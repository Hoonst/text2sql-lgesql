#coding=utf8
import sys, os, time, json, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import Namespace
from utils.args import init_args
from utils.hyperparams import hyperparam_path
from utils.initialization import *
from utils.example import Example
from utils.batch import Batch
from utils.optimization import set_optimizer
from model.model_utils import Registrable
from model.model_constructor import *
from torch import nn
import wandb
import glob

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
exp_path = hyperparam_path(args)

logger = set_logger(exp_path, args.testing)
set_random_seed(args.seed)
device = set_torch_device(args.device)
logger.info("Initialization finished ...")
logger.info("Output path is %s" % (exp_path))
logger.info("Random seed is set to %d" % (args.seed))
logger.info("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

# load dataset and vocabulary
start_time = time.time()
if args.read_model_path and args.read_model_path != '.':
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    params.lazy_load = True
elif args.read_model_path == '.':
    params = args
# set up the grammar, transition system, evaluator, etc.
Example.configuration(plm=params.plm, method=params.model)
train_dataset, dev_dataset = Example.load_dataset('train'), Example.load_dataset('dev')
# loading dataset needs about 2 min
# if we apply dropedge here, then for whole process
# only limited number of examples are introduced

logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
sql_trans, evaluator = Example.trans, Example.evaluator
args.word_vocab, args.relation_num = len(Example.word_vocab), len(Example.relation_vocab)

# model init, set optimizer
model = Registrable.by_name('text2sql')(params, sql_trans).to(device)
# model = nn.DataParallel(model)

if args.read_model_path and args.read_model_path != '.':
    # check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
    # model.load_state_dict(check_point['model'])
    # logger.info("Load saved model from path: %s" % (args.read_model_path))

    checkpoint_path = sorted(glob.glob(args.read_model_path+'/*.bin'))[-1]
    check_point = torch.load(open(checkpoint_path, 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    logger.info("Load saved model from path: %s" % (args.read_model_path))

elif args.read_model_path == '.':
    json.dump(vars(params), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
    if params.plm is None:
        ratio = Example.word2vec.load_embeddings(model.encoder.input_layer.word_embed, Example.word_vocab, device=device)
        logger.info("Init model and word embedding layer with a coverage %.2f" % (ratio))
# logger.info(str(model))

def decode(choice, output_path, acc_type='sql', use_checker=False):
    assert acc_type in ['beam', 'ast', 'sql'] and choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    all_hyps = []
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False)
            hyps = model.parse(current_batch, args.beam_size)
            all_hyps.extend(hyps)
        acc = evaluator.acc(all_hyps, dataset, output_path, acc_type=acc_type, etype='match', use_checker=use_checker)
    torch.cuda.empty_cache()
    gc.collect()
    return acc

if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    logger.info('Total training steps: %d;\t Warmup steps: %d' % (num_training_steps, num_warmup_steps))
    optimizer, scheduler = set_optimizer(model, args, num_warmup_steps, num_training_steps)
    start_epoch, nsamples, best_result = 0, len(train_dataset), {'dev_acc': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size // args.grad_accumulate
    if args.read_model_path and args.read_model_path != '.':
        optimizer.load_state_dict(check_point['optim'])
        scheduler.load_state_dict(check_point['scheduler'])
        start_epoch = check_point['epoch'] + 1

    # import IPython; IPython.embed(); exit(1);
    logger.info('Start training ......')

    wandb.init(project='LGESQL Experiments', name=exp_path)
    for i in range(start_epoch, args.max_epoch + 1):

        start_time = time.time()
        epoch_loss, epoch_gp_loss, count = 0, 0, 0
        np.random.shuffle(train_index)
        model.train()
        '''
        step_size
        '''
        for j in range(0, nsamples, step_size):
            count += 1
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = Batch.from_example_list(cur_dataset, device, train=True, smoothing=args.smoothing)
            '''
            Batch.from_example_list
            batch.py
            > from_example_list
            > from_example_list_text2sql
            > from_example_list_base
            '''
            loss, gp_loss = model(current_batch) # see utils/batch.py for batch elements
            wandb.log({'loss': loss, 'gp_loss': gp_loss})
            epoch_loss += loss.item()
            epoch_gp_loss += gp_loss.item()
            # print("Minibatch loss: %.4f" % (loss.item()))
            loss += gp_loss

            loss.backward()
            if count == args.grad_accumulate or j + step_size >= nsamples:
                count = 0
                model.pad_embedding_grad_zero()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        logger.info('Training: \tEpoch: %d\tTime: %.4f\tTraining loss: %.4f/%.4f' % (i, time.time() - start_time, epoch_loss, epoch_gp_loss))
        torch.cuda.empty_cache()
        gc.collect()

        if i < args.eval_after_epoch: # avoid unnecessary evaluation
            continue

        elif i >= args.eval_after_epoch and i in list(range(args.eval_after_epoch, args.max_epoch+1, 20)): 
            start_time = time.time()
            dev_acc = decode('dev', os.path.join(exp_path, 'dev.iter' + str(i)), acc_type='sql')
            logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_acc))
            wandb.log({'dev_acc': dev_acc})
            if dev_acc > best_result['dev_acc']:
                best_result['dev_acc'], best_result['iter'] = dev_acc, i
                logger.info('NEW BEST MODEL: \tEpoch: %d\tDev acc: %.4f' % (i, dev_acc))

            i_length = 4
            zeros = '0' * (i_length - len(str(i)))
            new_i = zeros + str(i)

            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, open(os.path.join(exp_path, f'model_epoch_{new_i}.bin'), 'wb'))
                

    logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev acc: %.4f' % (best_result['iter'], best_result['dev_acc']))
    # check_point = torch.load(open(os.path.join(exp_path, 'model.bin'), 'rb'))
    # model.load_state_dict(check_point['model'])
    # dev_acc_beam = decode('dev', output_path=os.path.join(exp_path, 'dev.iter' + str(best_result['iter']) + '.beam' + str(args.beam_size)), acc_type='beam')
    # logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev acc/Beam acc: %.4f/%.4f' % (best_result['iter'], best_result['dev_acc'], dev_acc_beam))
else:
    # start_time = time.time()
    # train_acc = decode('train', output_path=os.path.join(args.read_model_path, 'train.eval'), acc_type='sql')
    # logger.info("Evaluation costs %.2fs ; Train dataset exact match acc is %.4f ." % (time.time() - start_time, train_acc))
    start_time = time.time()
    dev_acc = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval'), acc_type='sql')
    dev_acc_checker = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval.checker'), acc_type='sql', use_checker=True)
    dev_acc_beam = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval.beam' + str(args.beam_size)), acc_type='beam')
    logger.info("Evaluation costs %.2fs ; Dev dataset exact match/checker/beam acc is %.4f/%.4f/%.4f ." % (time.time() - start_time, dev_acc, dev_acc_checker, dev_acc_beam))

PAD = '[PAD]'
BOS = '[CLS]'
EOS = '[SEP]'
UNK = '[UNK]'

GRAMMAR_FILEPATH = 'asdl/sql/grammar/sql_asdl_v2.txt'
SCHEMA_TYPES = ['table', 'others', 'text', 'time', 'number', 'boolean']
MAX_RELATIVE_DIST = 2
# relations: type_1-type_2-rel_name, r represents reverse edge, b represents bidirectional edge
RELATIONS = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
    ['table-table-identity', 'table-table-fk', 'table-table-fkr', 'table-table-fkb'] + \
    ['column-column-identity', 'column-column-sametable', 'column-column-fk', 'column-column-fkr'] + \
    ['table-column-pk', 'column-table-pk', 'table-column-has', 'column-table-has'] + \
    ['question-column-exactmatch', 'question-column-partialmatch', 'question-column-nomatch', 'question-column-valuematch',
    'column-question-exactmatch', 'column-question-partialmatch', 'column-question-nomatch', 'column-question-valuematch'] + \
    ['question-table-exactmatch', 'question-table-partialmatch', 'question-table-nomatch',
    'table-question-exactmatch', 'table-question-partialmatch', 'table-question-nomatch'] + \
    ['question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic'] + \
    ['*-*-identity', '*-question-generic', 'question-*-generic', '*-table-generic', 'table-*-generic', '*-column-generic', 'column-*-generic']

'''
local_relations = 
{'question-table-exactmatch', 'table-*-generic', 'column-question-nomatch', 
'question-column-nomatch', 'table-column-pk', 'table-question-nomatch', 'question-column-partialmatch', 
'column-question-valuematch', 'question-column-exactmatch', 'question-*-generic', 
'question-question-dist-1', 'column-table-has', 'question-table-nomatch', 'column-column-fkr', 
'column-question-exactmatch', 'question-question-dist1', 'column-question-partialmatch', 
'*-question-generic', 'table-question-exactmatch', 'question-column-valuematch', 'column-column-fk', 
'table-column-has', 'table-question-partialmatch', 'question-table-partialmatch', 'column-table-pk'}

nonlocal_relations = {'column-*-generic', 'table-table-fkr', 'question-question-generic', '*-column-generic', 
'column-column-identity', '*-*-identity', '*-table-generic', 'table-column-generic', 'table-table-generic', 
'table-table-fk', 'question-question-dist2', 'question-question-dist-2', 'column-column-generic', 'column-column-sametable', 
'table-table-fkb', 'question-question-identity', 'column-table-generic', 'table-table-identity'}

'''


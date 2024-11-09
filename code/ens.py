import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sub1 = pd.read_csv('push1.csv').sort_values(['id']).reset_index(drop=True)
sub2 = pd.read_csv('push4.csv').sort_values(['id']).reset_index(drop=True)
sub3 = pd.read_csv('push5.csv').sort_values(['id']).reset_index(drop=True)
sub4 = pd.read_csv('push6.csv').sort_values(['id']).reset_index(drop=True)
sub5 = pd.read_csv('push7.csv').sort_values(['id']).reset_index(drop=True)
sub8 = pd.read_csv('push8.csv').sort_values(['id']).reset_index(drop=True)
MMscaler = MinMaxScaler()

p1 = MMscaler.fit_transform(sub1['score'].values.reshape(-1,1)).reshape(-1)
p2 = MMscaler.fit_transform(sub2['score'].values.reshape(-1,1)).reshape(-1)
p3 = MMscaler.fit_transform(sub3['score'].values.reshape(-1,1)).reshape(-1)
p4 = MMscaler.fit_transform(sub4['score'].values.reshape(-1,1)).reshape(-1)

p5 = MMscaler.fit_transform(sub5['score'].values.reshape(-1,1)).reshape(-1)
p8 = MMscaler.fit_transform(sub8['score'].values.reshape(-1,1)).reshape(-1)

pub = pd.read_csv('pushp.csv').sort_values(['id']).reset_index(drop=True)
ppub = MMscaler.fit_transform(pub['score'].values.reshape(-1,1)).reshape(-1)

submission = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/sample_submission.csv').sort_values(['id']).reset_index(drop=True)
submission['score'] = (p1*0.401+(p2+p5+1.5*p8)/3.5*0.38+p3*0.199+p4*0.02)*0.751+ppub*0.249
submission[['id', 'score']].to_csv('submission.csv', index=False)
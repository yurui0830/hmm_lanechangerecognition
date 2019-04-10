from hmmtest import hmm_test

print(hmm_test('rightlc', training_set='5s', slot='1s', n_feature=3))
print(hmm_test('leftlc', training_set='5s', slot='1s', n_feature=3))
print(hmm_test('lk', training_set='5s', slot='1s', n_feature=3))

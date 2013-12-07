#!/usr/bin/env python

import fastdict

f_dict = fastdict.FastDict()
f_dict.set(123, 456, 'vec0')
print f_dict.size()

print f_dict.get(123)
print f_dict.get(123)[0]
print f_dict.get(123)[0].first
print f_dict.get(123)[0].second
 
f_dict.append(123, 678, 'vec1')
print f_dict.size()

print f_dict.get(123)
print f_dict.get(123)[0]
print f_dict.get(123)[0].first
print f_dict.get(123)[0].second
print f_dict.get(123)[1]
print f_dict.get(123)[1].first
print f_dict.get(123)[1].second

for ele in f_dict.get(123):
    print ele
    print ele.first
    print ele.second

 

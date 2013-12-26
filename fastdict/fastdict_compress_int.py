#!/usr/bin/env python

import fastdict

f_dict = fastdict.FastCompressIntDict(32)
f_dict.set(123, 456, 0)
print f_dict.size()

print f_dict.get(123)
print f_dict.get(123)[0]
print f_dict.get(123)[0].first
print f_dict.get(123)[0].second
 
f_dict.append(123, 678, 1)
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

f_dict.set(456, 789, 2)

print f_dict.keys()

for key in f_dict.keys():
    print "key: " + str(key)

f_dict.set_keydimensions([1, 2, 3])

fastdict.save_compress_int("test.dict", f_dict)

f_dict = fastdict.FastCompressIntDict(32)

print f_dict.size()

fastdict.load_compress_int("test.dict", f_dict)

print f_dict.size()
 
f_dict_merge_source = fastdict.FastCompressIntDict(32)

f_dict_merge_source.set(789, 123, 3)

print f_dict_merge_source.size()
 
for key in f_dict_merge_source.keys():
    print "key: " + str(key)

f_dict.merge(f_dict_merge_source)

print "merged: "
print f_dict.size()

f_dict_merge_source.clear()

print f_dict_merge_source.size()
 
for ele in f_dict.get(789):
    print ele
    print ele.first
    print ele.second
 
for ele in f_dict.get(123):
    print ele
    print ele.first
    print ele.second

key_dimensions = []

print f_dict.get_keydimensions(key_dimensions)

print key_dimensions

print f_dict.exist(123)
print f_dict.exist(12345)

for ele in f_dict.get(12345):
    print ele
    print ele.first
    print ele.second


 

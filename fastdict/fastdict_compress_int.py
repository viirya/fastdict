#!/usr/bin/env python

import fastdict
import sys
import struct

f_dict = fastdict.FastCompressIntDict(16)
f_dict.set(123, 6794572984750169060, 0)
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

for key in f_dict.mget([123, 456]):
    print key.first
    print key.second

f_dict.set_keydimensions([1, 2, 3])

fastdict.save_compress_int("test.dict", f_dict)

f_dict = fastdict.FastCompressIntDict(8)

print f_dict.size()

fastdict.load_compress_int("test.dict", f_dict)

print f_dict.size()
 
f_dict_merge_source = fastdict.FastCompressIntDict(8)

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

f_dict.go_index()

cols = f_dict.get_cols(123)
col_count = 0
for column in cols.first:
    print "col: " + str(col_count)
    for bit_count in column:
        print bit_count
    col_count += 1

for image_id in cols.second:
    print image_id


print f_dict.size()
 
fastdict.save_compress_int("test.dict", f_dict)

f_dict = fastdict.FastCompressIntDict(8)
print f_dict.size()

cols = f_dict.get_cols(123)
col_count = 0
for column in cols.first:
    print "col: " + str(col_count)
    for bit_count in column:
        print bit_count
    col_count += 1

for image_id in cols.second:
    print image_id

fastdict.load_compress_int("test.dict", f_dict)

cols = f_dict.get_cols(123)
col_count = 0
for column in cols.first:
    print "col: " + str(col_count)
    for bit_count in column:
        print bit_count
    col_count += 1

for image_id in cols.second:
    print image_id

# get_binary_codes should be called before runtime dict initialization
binary_codes = f_dict.get_binary_codes(123)
for code in binary_codes.first:
    print "code: " + str(code)

print "mget binary codes:"
binary_codes = f_dict.mget_binary_codes([123, 456])
for code in binary_codes.first:
    print "code: " + str(code)
 

# initialze runtime dict
print "init runtime dict..."
f_dict.init_runtime_dict()
print "done."

print "buffer:"
cols_buffer = f_dict.get_cols_as_buffer(123)
print len(cols_buffer)
print cols_buffer
index = 0
for buffers in cols_buffer:
    print index
    print "len: " + str(len(buffers))
    for i in range(0, len(buffers) / 8):
        data = ''
        for j in range(i * 8, i * 8 + 8):
            data = data + buffers[j]
        print data
        print struct.unpack('Q', data)
    index += 1
 
print "get multiple buffers:"
cols_buffers = f_dict.mget_cols_as_buffer([123, 456])
print len(cols_buffers)
print cols_buffers

for cols_buffer in cols_buffers:
    index = 0
    for buffers in cols_buffer:
        print index
        for i in range(0, len(buffers) / 8):
            data = ''
            for j in range(i * 8, i * 8 + 8):
                data = data + buffers[j]
            print data
            print struct.unpack('Q', data)
        index += 1


## VLQ base64

vlq_dict = fastdict.FastCompressUInt32IntDict(8)
print "Test VLQ base64"
print vlq_dict.base64VLQ_encode(123123)
for val in vlq_dict.base64VLQ_decode(vlq_dict.base64VLQ_encode(123123)):
    print val

for val in vlq_dict.base64VLQ_decode('AAgBC'):
    print val

vlq_dict.set(123, 6794572984750169060, 0)
vlq_dict.append(123, 678, 1)
print vlq_dict.size()

vlq_dict.go_index() # compress
vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict
cols = vlq_dict.get_VLQ_base64_cols(123)
print cols.first
print cols.second

for string in cols.first:
    print string
    print "decode:"
    for val in vlq_dict.base64VLQ_decode(string):
        print val
for image_id in cols.second:
    print image_id

print "dict status: " + str(vlq_dict.get_dict_status())

print "cpu-based uncompression for VLQ base64"
binary_codes = vlq_dict.get_VLQ_base64_binary_codes(123)
for code in binary_codes.first:    
    print "code: " + str(code)

print "mget VQL base64 binary codes:"
binary_codes = vlq_dict.mget_VLQ_base64_binary_codes([123])
for code in binary_codes.first:    
    print "code: " + str(code)
 

# init runtime VLQ base64 dict
vlq_dict.init_runtime_VLQ_base64_dict()

print "dict status: " + str(vlq_dict.get_dict_status())

print "buffer:"
VLQ_cols_buffer = vlq_dict.get_VLQ_base64_cols_as_buffer(123)
print len(VLQ_cols_buffer)
print VLQ_cols_buffer

index = 0
for buffers in VLQ_cols_buffer:    
    print index
    for i in range(0, len(buffers)):
        print struct.unpack('c', buffers[i])
        print buffers[i]
    index += 1
    
print "multiple buffer:"
VLQ_cols_buffers = vlq_dict.mget_VLQ_base64_cols_as_buffer([123])
print len(VLQ_cols_buffers)
print VLQ_cols_buffers

for VLQ_cols_buffer in VLQ_cols_buffers:
    index = 0
    for buffers in VLQ_cols_buffer:    
        print index
        for i in range(0, len(buffers)):
            print struct.unpack('c', buffers[i])
            print buffers[i]
        index += 1
 
 


 


#!/usr/bin/env python

import unittest
import fastdict
import sys
import struct

class TestFastCompressUInt32IntDict(unittest.TestCase):

    def setUp(self):
        self.dimension = 16
 
    def test_size(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)
        self.assertEqual(f_dict.size(), 1)

        f_dict.append(123, 678, 1)
        self.assertEqual(f_dict.size(), 1)
 
        f_dict.set(456, 6794572984750169060, 0)
        self.assertEqual(f_dict.size(), 2)
 
    def test_setandget(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)

        self.assertEqual(f_dict.get(123)[0].first, 6794572984750169060)
        self.assertEqual(f_dict.get(123)[0].second, 0)

        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])


        self.assertEqual(f_dict.get(123)[0].first, 6794572984750169060)
        self.assertEqual(f_dict.get(123)[0].second, 0)
 
        self.assertEqual(f_dict.get(123)[1].first, 678)
        self.assertEqual(f_dict.get(123)[1].second, 1)
        
        self.assertEqual(f_dict.get(123)[2].first, 456)
        self.assertEqual(f_dict.get(123)[2].second, 3)
 
        self.assertEqual(f_dict.get(123)[3].first, 123123)
        self.assertEqual(f_dict.get(123)[3].second, 4)

        f_dict.set(456, 789, 2)

        self.assertEqual(f_dict.get(456)[0].first, 789)
        self.assertEqual(f_dict.get(456)[0].second, 2)

        multple_gets = f_dict.mget([123, 456])
        self.assertEqual(multple_gets[0].first, 6794572984750169060)
        self.assertEqual(multple_gets[0].second, 0)
        self.assertEqual(multple_gets[3].first, 123123)
        self.assertEqual(multple_gets[3].second, 4)

        self.assertTrue(f_dict.exist(123))
        self.assertFalse(f_dict.exist(12345))
 
    def test_getkeys(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)    
        f_dict.set(123, 78912893, 0)
        f_dict.set(456, 789, 1)

        keys = f_dict.keys()
        self.assertEqual(keys[0], 123)
        self.assertEqual(keys[1], 456)
        
    def test_saveandload(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)    
        f_dict.set(123, 78912893, 0)
        f_dict.set(456, 789, 1)

        fastdict.save_compress_uint32_int("test.dict", f_dict)
        another_f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        fastdict.load_compress_uint32_int("test.dict", another_f_dict)

        self.assertEqual(another_f_dict.size(), 2)

    def test_keydimensions(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        f_dict.set_keydimensions([1, 2, 3])

        keydimensions = []
        f_dict.get_keydimensions(keydimensions)
        self.assertEqual(keydimensions, [1, 2, 3])

        fastdict.save_compress_uint32_int("test.dict", f_dict)
        another_f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        fastdict.load_compress_uint32_int("test.dict", another_f_dict)

        keydimensions = []
        another_f_dict.get_keydimensions(keydimensions)
        self.assertEqual(keydimensions, [1, 2, 3])


    def test_merge(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        f_dict.set(123, 123123, 0)

        f_dict_merge_source = fastdict.FastCompressUInt32IntDict(self.dimension)
        f_dict_merge_source.set(789, 123, 1)

        self.assertEqual(f_dict.size(), 1)
        self.assertEqual(f_dict.get(789)[0].first, 0)
        self.assertEqual(f_dict.get(789)[0].second, 0)

        f_dict.merge(f_dict_merge_source)

        self.assertEqual(f_dict.size(), 2)
        self.assertEqual(f_dict.get(789)[0].first, 123)
        self.assertEqual(f_dict.get(789)[0].second, 1)

        f_dict_merge_source.clear()
        self.assertEqual(f_dict_merge_source.size(), 0)

    def test_compress(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)

        self.assertEqual(f_dict.get(123)[0].first, 6794572984750169060)
        self.assertEqual(f_dict.get(123)[0].second, 0)

        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])
        f_dict.set(456, 789, 2)
        f_dict.set(789, 123, 3)

        f_dict.go_index()
        cols = f_dict.get_cols(123)
        bitcounts = []
        for column in cols.first:
            for bit_count in column:
                bitcounts.append(bit_count)
        self.assertEqual(bitcounts[0], 2)
        self.assertEqual(bitcounts[1], 1)
        self.assertEqual(bitcounts[2], 1)
        self.assertEqual(bitcounts[3], 0)
        self.assertEqual(bitcounts[len(bitcounts) - 2], 4)
        self.assertEqual(bitcounts[len(bitcounts) - 1], 0)

        ids = []
        for image_id in cols.second:
            ids.append(image_id)

        self.assertEqual(ids, [3, 1, 4, 0])
 
    def test_getbinarycodes(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)
        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])
        f_dict.set(456, 789, 2)
        f_dict.set(789, 123, 3)

        f_dict.go_index()

        # get_binary_codes only works before runtime dict initiated
        binary_codes = f_dict.get_binary_codes(123)
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [456, 678, 123123, 6794572984750169060])

        binary_codes = f_dict.mget_binary_codes([123, 456])
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [456, 678, 123123, 6794572984750169060, 789])
 
    def test_runtimedict(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)
        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])
        f_dict.set(456, 789, 2)
        f_dict.set(789, 123, 3)

        f_dict.go_index()
        f_dict.init_runtime_dict()

        cols_buffer = f_dict.get_cols_as_buffer(123)
        # 64 columns 
        self.assertEqual(len(cols_buffer), 64)
        index = 0
        for buffers in cols_buffer:
            if index == 0:
                self.assertEqual(len(buffers), 16)
            for i in range(0, len(buffers) / 4):
                data = ''
                for j in range(i * 4, i * 4 + 4):
                    data = data + buffers[j]
                data = struct.unpack('I', data)
                if index == 0:
                    if i == 0:
                        self.assertEqual(data[0], 2)
                    if i == 1:
                        self.assertEqual(data[0], 1)
            index += 1
 
        cols_buffers = f_dict.mget_cols_as_buffer([123, 456])
        self.assertEqual(len(cols_buffers), 2)

        buffer_index = 0
        for cols_buffer in cols_buffers:
            index = 0
            for buffers in cols_buffer:
                for i in range(0, len(buffers) / 4):
                    data = ''
                    for j in range(i * 4, i * 4 + 4):
                        data = data + buffers[j]
                    data = struct.unpack('I', data)
                    if index == 63 and buffer_index == 1:
                        if i == 0:
                            self.assertEqual(data[0], 1)
                        if i == 1:
                            self.assertEqual(data[0], 0)
                index += 1
            buffer_index += 1
 
    def test_runtime_python_dict(self):
        f_dict = fastdict.FastCompressUInt32IntDict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)
        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])
        f_dict.set(456, 789, 2)
        f_dict.set(789, 123, 3)

        f_dict.go_index()
        f_dict.init_runtime_python_dict()

        cols_buffer = f_dict.get_python_cols_as_buffer(123)
        # 64 columns 
        self.assertEqual(len(cols_buffer), 64)
        index = 0
        for buffers in cols_buffer:
            if index == 0:
                self.assertEqual(len(buffers), 16)
            for i in range(0, len(buffers) / 4):
                data = ''
                for j in range(i * 4, i * 4 + 4):
                    data = data + buffers[j]
                data = struct.unpack('I', data)
                if index == 0:
                    if i == 0:
                        self.assertEqual(data[0], 2)
                    if i == 1:
                        self.assertEqual(data[0], 1)
            index += 1
 
        cols_buffers = f_dict.mget_python_cols_as_buffer([123, 456])
        self.assertEqual(len(cols_buffers), 2)

        buffer_index = 0
        for cols_buffer in cols_buffers:
            index = 0
            for buffers in cols_buffer:
                for i in range(0, len(buffers) / 4):
                    data = ''
                    for j in range(i * 4, i * 4 + 4):
                        data = data + buffers[j]
                    data = struct.unpack('I', data)
                    if index == 63 and buffer_index == 1:
                        if i == 0:
                            self.assertEqual(data[0], 1)
                        if i == 1:
                            self.assertEqual(data[0], 0)
                index += 1
            buffer_index += 1
 
    def test_VLQ_base64(self):
        vlq_dict = fastdict.FastCompressUInt32IntDict(8)
        self.assertEqual(vlq_dict.base64VLQ_encode(123123), 'zn4D')
        for val in vlq_dict.base64VLQ_decode(vlq_dict.base64VLQ_encode(123123)):
            self.assertEqual(val, 123123)

        vals = []
        for val in vlq_dict.base64VLQ_decode('AAgBC'):
            vals.append(val)
        self.assertEqual(vals, [0, 0, 32, 2])
 
    def test_VLQ_base64_dict(self):
        vlq_dict = fastdict.FastCompressUInt32IntDict(8)
        vlq_dict.set(123, 6794572984750169060, 0)
        vlq_dict.append(123, 678, 1)
        self.assertEqual(vlq_dict.size(), 1)

        vlq_dict.go_index() # compress
        vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict

        cols = vlq_dict.get_VLQ_base64_cols(123)
        strings = []
        decodes = []
        for string in cols.first:
            strings.append(string)
            for val in vlq_dict.base64VLQ_decode(string):
                decodes.append(val)

        self.assertEqual(strings[0], 'CA')
        self.assertEqual(decodes[0], 2)
        self.assertEqual(decodes[1], 0)

        image_ids = []
        for image_id in cols.second:
            image_ids.append(image_id)
        self.assertEqual(image_ids, [1, 0])

        self.assertEqual(vlq_dict.get_dict_status(), 1)

    def test_CPU_based_uncompress_VLQ_base64(self):
        vlq_dict = fastdict.FastCompressUInt32IntDict(8)
        vlq_dict.set(123, 6794572984750169060, 0)
        vlq_dict.append(123, 678, 1)
        self.assertEqual(vlq_dict.size(), 1)

        vlq_dict.go_index() # compress
        vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict

        binary_codes = vlq_dict.get_VLQ_base64_binary_codes(123)
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [678, 6794572984750169060])

        binary_codes = vlq_dict.mget_VLQ_base64_binary_codes([123])
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [678, 6794572984750169060])
 
    def test_VLQ_base64_runtimedict(self):
        vlq_dict = fastdict.FastCompressUInt32IntDict(8)
        vlq_dict.set(123, 6794572984750169060, 0)
        vlq_dict.append(123, 678, 1)
        self.assertEqual(vlq_dict.size(), 1)

        vlq_dict.go_index() # compress
        vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict

        # init runtime VLQ base64 dict
        vlq_dict.init_runtime_VLQ_base64_dict()

        self.assertEqual(vlq_dict.get_dict_status(), 3)

        VLQ_cols_buffer = vlq_dict.get_VLQ_base64_cols_as_buffer(123)
        # 64 columns
        self.assertEqual(len(VLQ_cols_buffer), 64)

        index = 0
        for buffers in VLQ_cols_buffer:    
            for i in range(0, len(buffers)):
                if index == 0:
                    if i == 0:
                        self.assertEqual(struct.unpack('c', buffers[i])[0], 'C')
                        self.assertEqual(buffers[i], 'C') 
                    if i == 1:
                        self.assertEqual(struct.unpack('c', buffers[i])[0], 'A')
                        self.assertEqual(buffers[i], 'A') 
            index += 1
            
        VLQ_cols_buffers = vlq_dict.mget_VLQ_base64_cols_as_buffer([123])
        self.assertEqual(len(VLQ_cols_buffers), 1)
       
        VLQ_cols_buffer_index = 0 
        for VLQ_cols_buffer in VLQ_cols_buffers:
            index = 0
            for buffers in VLQ_cols_buffer:    
                for i in range(0, len(buffers)):
                    if index == 0:
                        if i == 0:
                            self.assertEqual(struct.unpack('c', buffers[i])[0], 'C')
                            self.assertEqual(buffers[i], 'C') 
                        if i == 1:
                            self.assertEqual(struct.unpack('c', buffers[i])[0], 'A')
                            self.assertEqual(buffers[i], 'A') 
                index += 1
            VLQ_cols_buffer_index += 1
 
class TestFastCompressUInt32Int8Dict(unittest.TestCase):

    def setUp(self):
        self.dimension = 16
 
    def test_size(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)
        self.assertEqual(f_dict.size(), 1)

        f_dict.append(123, 678, 1)
        self.assertEqual(f_dict.size(), 1)
 
        f_dict.set(456, 6794572984750169060, 0)
        self.assertEqual(f_dict.size(), 2)
 
    def test_setandget(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)

        self.assertEqual(f_dict.get(123)[0].first, 6794572984750169060)
        self.assertEqual(f_dict.get(123)[0].second, 0)

        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])


        self.assertEqual(f_dict.get(123)[0].first, 6794572984750169060)
        self.assertEqual(f_dict.get(123)[0].second, 0)
 
        self.assertEqual(f_dict.get(123)[1].first, 678)
        self.assertEqual(f_dict.get(123)[1].second, 1)
        
        self.assertEqual(f_dict.get(123)[2].first, 456)
        self.assertEqual(f_dict.get(123)[2].second, 3)
 
        self.assertEqual(f_dict.get(123)[3].first, 123123)
        self.assertEqual(f_dict.get(123)[3].second, 4)

        f_dict.set(456, 789, 2)

        self.assertEqual(f_dict.get(456)[0].first, 789)
        self.assertEqual(f_dict.get(456)[0].second, 2)

        multple_gets = f_dict.mget([123, 456])
        self.assertEqual(multple_gets[0].first, 6794572984750169060)
        self.assertEqual(multple_gets[0].second, 0)
        self.assertEqual(multple_gets[3].first, 123123)
        self.assertEqual(multple_gets[3].second, 4)

        self.assertTrue(f_dict.exist(123))
        self.assertFalse(f_dict.exist(12345))
 
    def test_getkeys(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)    
        f_dict.set(123, 78912893, 0)
        f_dict.set(456, 789, 1)

        keys = f_dict.keys()
        self.assertEqual(keys[0], 123)
        self.assertEqual(keys[1], 456)
        
    def test_saveandload(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)    
        f_dict.set(123, 78912893, 0)
        f_dict.set(456, 789, 1)

        fastdict.save_compress_uint32_int8("test.dict", f_dict)
        another_f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        fastdict.load_compress_uint32_int8("test.dict", another_f_dict)

        self.assertEqual(another_f_dict.size(), 2)

    def test_keydimensions(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        f_dict.set_keydimensions([1, 2, 3])

        keydimensions = []
        f_dict.get_keydimensions(keydimensions)
        self.assertEqual(keydimensions, [1, 2, 3])

        fastdict.save_compress_uint32_int8("test.dict", f_dict)
        another_f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        fastdict.load_compress_uint32_int8("test.dict", another_f_dict)

        keydimensions = []
        another_f_dict.get_keydimensions(keydimensions)
        self.assertEqual(keydimensions, [1, 2, 3])


    def test_merge(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        f_dict.set(123, 123123, 0)

        f_dict_merge_source = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        f_dict_merge_source.set(789, 123, 1)

        self.assertEqual(f_dict.size(), 1)
        self.assertEqual(f_dict.get(789)[0].first, 0)
        self.assertEqual(f_dict.get(789)[0].second, 0)

        f_dict.merge(f_dict_merge_source)

        self.assertEqual(f_dict.size(), 2)
        self.assertEqual(f_dict.get(789)[0].first, 123)
        self.assertEqual(f_dict.get(789)[0].second, 1)

        f_dict_merge_source.clear()
        self.assertEqual(f_dict_merge_source.size(), 0)

    def test_compress(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)

        self.assertEqual(f_dict.get(123)[0].first, 6794572984750169060)
        self.assertEqual(f_dict.get(123)[0].second, 0)

        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])
        f_dict.set(456, 789, 2)
        f_dict.set(789, 123, 3)

        f_dict.go_index()
        cols = f_dict.get_cols(123)
        bitcounts = []
        for column in cols.first:
            for bit_count in column:
                bitcounts.append(bit_count)
        self.assertEqual(bitcounts[0], 2)
        self.assertEqual(bitcounts[1], 1)
        self.assertEqual(bitcounts[2], 1)
        self.assertEqual(bitcounts[3], 0)
        self.assertEqual(bitcounts[len(bitcounts) - 2], 4)
        self.assertEqual(bitcounts[len(bitcounts) - 1], 0)

        ids = []
        for image_id in cols.second:
            ids.append(image_id)

        self.assertEqual(ids, [3, 1, 4, 0])
 
    def test_getbinarycodes(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)
        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])
        f_dict.set(456, 789, 2)
        f_dict.set(789, 123, 3)

        f_dict.go_index()

        # get_binary_codes only works before runtime dict initiated
        binary_codes = f_dict.get_binary_codes(123)
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [456, 678, 123123, 6794572984750169060])

        binary_codes = f_dict.mget_binary_codes([123, 456])
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [456, 678, 123123, 6794572984750169060, 789])
 
    def test_runtimedict(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)
        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])
        f_dict.set(456, 789, 2)
        f_dict.set(789, 123, 3)

        f_dict.go_index()
        f_dict.init_runtime_dict()

        cols_buffer = f_dict.get_cols_as_buffer(123)
        # 64 columns 
        self.assertEqual(len(cols_buffer), 64)
        index = 0
        for buffers in cols_buffer:
            if index == 0:
                self.assertEqual(len(buffers), 16)
            for i in range(0, len(buffers) / 4):
                data = ''
                for j in range(i * 4, i * 4 + 4):
                    data = data + buffers[j]
                data = struct.unpack('I', data)
                if index == 0:
                    if i == 0:
                        self.assertEqual(data[0], 2)
                    if i == 1:
                        self.assertEqual(data[0], 1)
            index += 1
 
        cols_buffers = f_dict.mget_cols_as_buffer([123, 456])
        self.assertEqual(len(cols_buffers), 2)

        buffer_index = 0
        for cols_buffer in cols_buffers:
            index = 0
            for buffers in cols_buffer:
                for i in range(0, len(buffers) / 4):
                    data = ''
                    for j in range(i * 4, i * 4 + 4):
                        data = data + buffers[j]
                    data = struct.unpack('I', data)
                    if index == 63 and buffer_index == 1:
                        if i == 0:
                            self.assertEqual(data[0], 1)
                        if i == 1:
                            self.assertEqual(data[0], 0)
                index += 1
            buffer_index += 1
 
    def test_runtime_python_dict(self):
        f_dict = fastdict.FastCompressUInt32Int8Dict(self.dimension)
        f_dict.set(123, 6794572984750169060, 0)
        f_dict.append(123, 678, 1)
        f_dict.batch_append([123, 123], [456, 123123], [3, 4])
        f_dict.set(456, 789, 2)
        f_dict.set(789, 123, 3)

        f_dict.go_index()
        f_dict.init_runtime_python_dict()

        cols_buffer = f_dict.get_python_cols_as_buffer(123)
        # 64 columns 
        self.assertEqual(len(cols_buffer), 64)
        index = 0
        for buffers in cols_buffer:
            if index == 0:
                self.assertEqual(len(buffers), 16)
            for i in range(0, len(buffers) / 4):
                data = ''
                for j in range(i * 4, i * 4 + 4):
                    data = data + buffers[j]
                data = struct.unpack('I', data)
                if index == 0:
                    if i == 0:
                        self.assertEqual(data[0], 2)
                    if i == 1:
                        self.assertEqual(data[0], 1)
            index += 1
 
        cols_buffers = f_dict.mget_python_cols_as_buffer([123, 456])
        self.assertEqual(len(cols_buffers), 2)

        buffer_index = 0
        for cols_buffer in cols_buffers:
            index = 0
            for buffers in cols_buffer:
                for i in range(0, len(buffers) / 4):
                    data = ''
                    for j in range(i * 4, i * 4 + 4):
                        data = data + buffers[j]
                    data = struct.unpack('I', data)
                    if index == 63 and buffer_index == 1:
                        if i == 0:
                            self.assertEqual(data[0], 1)
                        if i == 1:
                            self.assertEqual(data[0], 0)
                index += 1
            buffer_index += 1
 
    def test_VLQ_base64(self):
        vlq_dict = fastdict.FastCompressUInt32Int8Dict(8)
        self.assertEqual(vlq_dict.base64VLQ_encode(123123), 'zn4D')
        for val in vlq_dict.base64VLQ_decode(vlq_dict.base64VLQ_encode(123123)):
            self.assertEqual(val, 123123)

        vals = []
        for val in vlq_dict.base64VLQ_decode('AAgBC'):
            vals.append(val)
        self.assertEqual(vals, [0, 0, 32, 2])
 
    def test_VLQ_base64_dict(self):
        vlq_dict = fastdict.FastCompressUInt32Int8Dict(8)
        vlq_dict.set(123, 6794572984750169060, 0)
        vlq_dict.append(123, 678, 1)
        self.assertEqual(vlq_dict.size(), 1)

        vlq_dict.go_index() # compress
        vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict

        cols = vlq_dict.get_VLQ_base64_cols(123)
        strings = []
        decodes = []
        for string in cols.first:
            strings.append(string)
            for val in vlq_dict.base64VLQ_decode(string):
                decodes.append(val)

        self.assertEqual(strings[0], 'CA')
        self.assertEqual(decodes[0], 2)
        self.assertEqual(decodes[1], 0)

        image_ids = []
        for image_id in cols.second:
            image_ids.append(image_id)
        self.assertEqual(image_ids, [1, 0])

        self.assertEqual(vlq_dict.get_dict_status(), 1)

    def test_CPU_based_uncompress_VLQ_base64(self):
        vlq_dict = fastdict.FastCompressUInt32Int8Dict(8)
        vlq_dict.set(123, 6794572984750169060, 0)
        vlq_dict.append(123, 678, 1)
        self.assertEqual(vlq_dict.size(), 1)

        vlq_dict.go_index() # compress
        vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict

        binary_codes = vlq_dict.get_VLQ_base64_binary_codes(123)
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [678, 6794572984750169060])

        binary_codes = vlq_dict.mget_VLQ_base64_binary_codes([123])
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [678, 6794572984750169060])
 
    def test_VLQ_base64_runtimedict(self):
        vlq_dict = fastdict.FastCompressUInt32Int8Dict(8)
        vlq_dict.set(123, 6794572984750169060, 0)
        vlq_dict.append(123, 678, 1)
        self.assertEqual(vlq_dict.size(), 1)

        vlq_dict.go_index() # compress
        vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict

        # init runtime VLQ base64 dict
        vlq_dict.init_runtime_VLQ_base64_dict()

        self.assertEqual(vlq_dict.get_dict_status(), 3)

        VLQ_cols_buffer = vlq_dict.get_VLQ_base64_cols_as_buffer(123)
        # 64 columns
        self.assertEqual(len(VLQ_cols_buffer), 64)

        index = 0
        for buffers in VLQ_cols_buffer:    
            for i in range(0, len(buffers)):
                if index == 0:
                    if i == 0:
                        self.assertEqual(struct.unpack('c', buffers[i])[0], 'C')
                        self.assertEqual(buffers[i], 'C') 
                    if i == 1:
                        self.assertEqual(struct.unpack('c', buffers[i])[0], 'A')
                        self.assertEqual(buffers[i], 'A') 
            index += 1
            
        VLQ_cols_buffers = vlq_dict.mget_VLQ_base64_cols_as_buffer([123])
        self.assertEqual(len(VLQ_cols_buffers), 1)
       
        VLQ_cols_buffer_index = 0 
        for VLQ_cols_buffer in VLQ_cols_buffers:
            index = 0
            for buffers in VLQ_cols_buffer:    
                for i in range(0, len(buffers)):
                    if index == 0:
                        if i == 0:
                            self.assertEqual(struct.unpack('c', buffers[i])[0], 'C')
                            self.assertEqual(buffers[i], 'C') 
                        if i == 1:
                            self.assertEqual(struct.unpack('c', buffers[i])[0], 'A')
                            self.assertEqual(buffers[i], 'A') 
                index += 1
            VLQ_cols_buffer_index += 1
 

class TestFastCompressUInt32StringDict(unittest.TestCase):

    def setUp(self):
        self.dimension = 16
 
    def test_size(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        f_dict.set(123, 6794572984750169060, "0")
        self.assertEqual(f_dict.size(), 1)

        f_dict.append(123, 678, "1")
        self.assertEqual(f_dict.size(), 1)
 
        f_dict.set(456, 6794572984750169060, "0")
        self.assertEqual(f_dict.size(), 2)
 
    def test_setandget(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        f_dict.set(123, 6794572984750169060, "0")

        self.assertEqual(f_dict.get(123)[0].first, 6794572984750169060)
        self.assertEqual(f_dict.get(123)[0].second, "0")

        f_dict.append(123, 678, "1")
        f_dict.batch_append([123, 123], [456, 123123], ["3", "4"])


        self.assertEqual(f_dict.get(123)[0].first, 6794572984750169060)
        self.assertEqual(f_dict.get(123)[0].second, "0")
 
        self.assertEqual(f_dict.get(123)[1].first, 678)
        self.assertEqual(f_dict.get(123)[1].second, "1")
        
        self.assertEqual(f_dict.get(123)[2].first, 456)
        self.assertEqual(f_dict.get(123)[2].second, "3")
 
        self.assertEqual(f_dict.get(123)[3].first, 123123)
        self.assertEqual(f_dict.get(123)[3].second, "4")

        f_dict.set(456, 789, "2")

        self.assertEqual(f_dict.get(456)[0].first, 789)
        self.assertEqual(f_dict.get(456)[0].second, "2")

        multple_gets = f_dict.mget([123, 456])
        self.assertEqual(multple_gets[0].first, 6794572984750169060)
        self.assertEqual(multple_gets[0].second, "0")
        self.assertEqual(multple_gets[3].first, 123123)
        self.assertEqual(multple_gets[3].second, "4")

        self.assertTrue(f_dict.exist(123))
        self.assertFalse(f_dict.exist(12345))
 
    def test_getkeys(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)    
        f_dict.set(123, 78912893, "0")
        f_dict.set(456, 789, "1")

        keys = f_dict.keys()
        self.assertEqual(keys[0], 123)
        self.assertEqual(keys[1], 456)
        
    def test_saveandload(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)    
        f_dict.set(123, 78912893, "0")
        f_dict.set(456, 789, "1")

        fastdict.save_compress_uint32_string("test.dict", f_dict)
        another_f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        fastdict.load_compress_uint32_string("test.dict", another_f_dict)

        self.assertEqual(another_f_dict.size(), 2)

    def test_keydimensions(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        f_dict.set_keydimensions([1, 2, 3])

        keydimensions = []
        f_dict.get_keydimensions(keydimensions)
        self.assertEqual(keydimensions, [1, 2, 3])

        fastdict.save_compress_uint32_string("test.dict", f_dict)
        another_f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        fastdict.load_compress_uint32_string("test.dict", another_f_dict)

        keydimensions = []
        another_f_dict.get_keydimensions(keydimensions)
        self.assertEqual(keydimensions, [1, 2, 3])


    def test_merge(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        f_dict.set(123, 123123, "0")

        f_dict_merge_source = fastdict.FastCompressUInt32StringDict(self.dimension)
        f_dict_merge_source.set(789, 123, "1")

        self.assertEqual(f_dict.size(), 1)
        self.assertEqual(f_dict.get(789)[0].first, 0)
        self.assertEqual(f_dict.get(789)[0].second, "")

        f_dict.merge(f_dict_merge_source)

        self.assertEqual(f_dict.size(), 2)
        self.assertEqual(f_dict.get(789)[0].first, 123)
        self.assertEqual(f_dict.get(789)[0].second, "1")

        f_dict_merge_source.clear()
        self.assertEqual(f_dict_merge_source.size(), 0)

    def test_compress(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        f_dict.set(123, 6794572984750169060, "0")

        self.assertEqual(f_dict.get(123)[0].first, 6794572984750169060)
        self.assertEqual(f_dict.get(123)[0].second, "0")

        f_dict.append(123, 678, "1")
        f_dict.batch_append([123, 123], [456, 123123], ["3", "4"])
        f_dict.set(456, 789, "2")
        f_dict.set(789, 123, "3")

        f_dict.go_index()
        cols = f_dict.get_cols(123)
        bitcounts = []
        for column in cols.first:
            for bit_count in column:
                bitcounts.append(bit_count)
        self.assertEqual(bitcounts[0], 2)
        self.assertEqual(bitcounts[1], 1)
        self.assertEqual(bitcounts[2], 1)
        self.assertEqual(bitcounts[3], 0)
        self.assertEqual(bitcounts[len(bitcounts) - 2], 4)
        self.assertEqual(bitcounts[len(bitcounts) - 1], 0)

        ids = []
        for image_id in cols.second:
            ids.append(image_id)

        self.assertEqual(ids, ["3", "1", "4", "0"])
 
    def test_getbinarycodes(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        f_dict.set(123, 6794572984750169060, "0")
        f_dict.append(123, 678, "1")
        f_dict.batch_append([123, 123], [456, 123123], ["3", "4"])
        f_dict.set(456, 789, "2")
        f_dict.set(789, 123, "3")

        f_dict.go_index()

        # get_binary_codes only works before runtime dict initiated
        binary_codes = f_dict.get_binary_codes(123)
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [456, 678, 123123, 6794572984750169060])

        binary_codes = f_dict.mget_binary_codes([123, 456])
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [456, 678, 123123, 6794572984750169060, 789])
 
    def test_runtimedict(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        f_dict.set(123, 6794572984750169060, "0")
        f_dict.append(123, 678, "1")
        f_dict.batch_append([123, 123], [456, 123123], ["3", "4"])
        f_dict.set(456, 789, "2")
        f_dict.set(789, 123, "3")

        f_dict.go_index()
        f_dict.init_runtime_dict()

        cols_buffer = f_dict.get_cols_as_buffer(123)
        # 64 columns 
        self.assertEqual(len(cols_buffer), 64)
        index = 0
        for buffers in cols_buffer:
            if index == 0:
                self.assertEqual(len(buffers), 16)
            for i in range(0, len(buffers) / 4):
                data = ''
                for j in range(i * 4, i * 4 + 4):
                    data = data + buffers[j]
                data = struct.unpack('I', data)
                if index == 0:
                    if i == 0:
                        self.assertEqual(data[0], 2)
                    if i == 1:
                        self.assertEqual(data[0], 1)
            index += 1
 
        cols_buffers = f_dict.mget_cols_as_buffer([123, 456])
        self.assertEqual(len(cols_buffers), 2)

        buffer_index = 0
        for cols_buffer in cols_buffers:
            index = 0
            for buffers in cols_buffer:
                for i in range(0, len(buffers) / 4):
                    data = ''
                    for j in range(i * 4, i * 4 + 4):
                        data = data + buffers[j]
                    data = struct.unpack('I', data)
                    if index == 63 and buffer_index == 1:
                        if i == 0:
                            self.assertEqual(data[0], 1)
                        if i == 1:
                            self.assertEqual(data[0], 0)
                index += 1
            buffer_index += 1
 
    def test_runtime_python_dict(self):
        f_dict = fastdict.FastCompressUInt32StringDict(self.dimension)
        f_dict.set(123, 6794572984750169060, "0")
        f_dict.append(123, 678, "1")
        f_dict.batch_append([123, 123], [456, 123123], ["3", "4"])
        f_dict.set(456, 789, "2")
        f_dict.set(789, 123, "3")

        f_dict.go_index()
        f_dict.init_runtime_python_dict()

        cols_buffer = f_dict.get_python_cols_as_buffer(123)
        # 64 columns 
        self.assertEqual(len(cols_buffer), 64)
        index = 0
        for buffers in cols_buffer:
            if index == 0:
                self.assertEqual(len(buffers), 16)
            for i in range(0, len(buffers) / 4):
                data = ''
                for j in range(i * 4, i * 4 + 4):
                    data = data + buffers[j]
                data = struct.unpack('I', data)
                if index == 0:
                    if i == 0:
                        self.assertEqual(data[0], 2)
                    if i == 1:
                        self.assertEqual(data[0], 1)
            index += 1
 
        cols_buffers = f_dict.mget_python_cols_as_buffer([123, 456])
        self.assertEqual(len(cols_buffers), 2)

        buffer_index = 0
        for cols_buffer in cols_buffers:
            index = 0
            for buffers in cols_buffer:
                for i in range(0, len(buffers) / 4):
                    data = ''
                    for j in range(i * 4, i * 4 + 4):
                        data = data + buffers[j]
                    data = struct.unpack('I', data)
                    if index == 63 and buffer_index == 1:
                        if i == 0:
                            self.assertEqual(data[0], 1)
                        if i == 1:
                            self.assertEqual(data[0], 0)
                index += 1
            buffer_index += 1
 
    def test_VLQ_base64(self):
        vlq_dict = fastdict.FastCompressUInt32StringDict(8)
        self.assertEqual(vlq_dict.base64VLQ_encode(123123), 'zn4D')
        for val in vlq_dict.base64VLQ_decode(vlq_dict.base64VLQ_encode(123123)):
            self.assertEqual(val, 123123)

        vals = []
        for val in vlq_dict.base64VLQ_decode('AAgBC'):
            vals.append(val)
        self.assertEqual(vals, [0, 0, 32, 2])


        encodeds = vlq_dict.NumberIdsToVLQ_base64([123123])
        self.assertEqual(encodeds[0], 'zn4D')
        decondeds = vlq_dict.VLQ_base64ToNumberIds(['zn4D'])
        self.assertEqual(decondeds[0], 123123)
 
    def test_VLQ_base64_dict(self):
        vlq_dict = fastdict.FastCompressUInt32StringDict(8)
        vlq_dict.set(123, 6794572984750169060, "0")
        vlq_dict.append(123, 678, "1")
        self.assertEqual(vlq_dict.size(), 1)

        vlq_dict.go_index() # compress
        vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict

        cols = vlq_dict.get_VLQ_base64_cols(123)
        strings = []
        decodes = []
        for string in cols.first:
            strings.append(string)
            for val in vlq_dict.base64VLQ_decode(string):
                decodes.append(val)

        self.assertEqual(strings[0], 'CA')
        self.assertEqual(decodes[0], 2)
        self.assertEqual(decodes[1], 0)

        image_ids = []
        for image_id in cols.second:
            image_ids.append(image_id)
        self.assertEqual(image_ids, ["1", "0"])

        self.assertEqual(vlq_dict.get_dict_status(), 1)

    def test_CPU_based_uncompress_VLQ_base64(self):
        vlq_dict = fastdict.FastCompressUInt32StringDict(8)
        vlq_dict.set(123, 6794572984750169060, "0")
        vlq_dict.append(123, 678, "1")
        self.assertEqual(vlq_dict.size(), 1)

        vlq_dict.go_index() # compress
        vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict

        binary_codes = vlq_dict.get_VLQ_base64_binary_codes(123)
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [678, 6794572984750169060])

        binary_codes = vlq_dict.mget_VLQ_base64_binary_codes([123])
        codes = []
        for code in binary_codes.first:
            codes.append(code)
        self.assertEqual(codes, [678, 6794572984750169060])
 
    def test_VLQ_base64_runtimedict(self):
        vlq_dict = fastdict.FastCompressUInt32StringDict(8)
        vlq_dict.set(123, 6794572984750169060, "0")
        vlq_dict.append(123, 678, "1")
        self.assertEqual(vlq_dict.size(), 1)

        vlq_dict.go_index() # compress
        vlq_dict.to_VLQ_base64_dict() # to VQL base64 dict

        # init runtime VLQ base64 dict
        vlq_dict.init_runtime_VLQ_base64_dict()

        self.assertEqual(vlq_dict.get_dict_status(), 3)

        VLQ_cols_buffer = vlq_dict.get_VLQ_base64_cols_as_buffer(123)
        # 64 columns
        self.assertEqual(len(VLQ_cols_buffer), 64)

        index = 0
        for buffers in VLQ_cols_buffer:    
            for i in range(0, len(buffers)):
                if index == 0:
                    if i == 0:
                        self.assertEqual(struct.unpack('c', buffers[i])[0], 'C')
                        self.assertEqual(buffers[i], 'C') 
                    if i == 1:
                        self.assertEqual(struct.unpack('c', buffers[i])[0], 'A')
                        self.assertEqual(buffers[i], 'A') 
            index += 1
            
        VLQ_cols_buffers = vlq_dict.mget_VLQ_base64_cols_as_buffer([123])
        self.assertEqual(len(VLQ_cols_buffers), 1)
       
        VLQ_cols_buffer_index = 0 
        for VLQ_cols_buffer in VLQ_cols_buffers:
            index = 0
            for buffers in VLQ_cols_buffer:    
                for i in range(0, len(buffers)):
                    if index == 0:
                        if i == 0:
                            self.assertEqual(struct.unpack('c', buffers[i])[0], 'C')
                            self.assertEqual(buffers[i], 'C') 
                        if i == 1:
                            self.assertEqual(struct.unpack('c', buffers[i])[0], 'A')
                            self.assertEqual(buffers[i], 'A') 
                index += 1
            VLQ_cols_buffer_index += 1
             
if __name__ == '__main__':
    unittest.main()


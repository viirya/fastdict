#include <string>
#include <map>
#include <vector>
#include <list>
#include <utility>
#include <algorithm>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <fstream>


#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>

#include <boost/python.hpp>

#include <boost/numpy.hpp>
 
void print_content_py_buffer(PyObject* obj) {

    Py_buffer *view;
    PyObject_GetBuffer(obj, view, PyBUF_SIMPLE);
    
    std::cout << view->format << "\n";
}

template <class T> 
void print_py_buffer(PyObject* obj) {

    Py_buffer *view;
    PyObject_GetBuffer(obj, view, PyBUF_SIMPLE);
    
    T* buf = (T*)(view->buf);
    std::cout << "len: " << view->len << "\n";
    for (int i = 0; i < view->len / 8; i++) {
        std::cout << (T)buf[i] << " ";
    }
    std::cout << "\n";
}
 
struct pyobject_to_python
{
    static PyObject* convert(PyObject const& obj_instance)
    { 
        PyObject* obj = (PyObject*)&obj_instance;

        return boost::python::incref(obj);
    }
};
 
struct pyobject_vector_to_python
{
    static PyObject* convert(std::vector<PyObject*> const& array)
    { 
        boost::python::list pylist;

        BOOST_FOREACH(PyObject* obj, array) {
            pylist.append(*obj);
        }

        return boost::python::incref(pylist.ptr());
    }
};


template <class IdType>
class FastDict
{

public:
    
    FastDict(uint8_t k_dim) : index_key_dimension(k_dim) {}

    friend class boost::serialization::access;

    void set(uint32_t key, uint64_t hash_key, IdType id) {

        std::vector<uint8_t> bool_key = actual_key(key);

        std::pair<uint64_t, IdType> element(hash_key, id);
        std::vector<std::pair<uint64_t, IdType> > element_list(1, element);
        dict[bool_key] = element_list;
    }

    void set_with_bool_key(std::vector<uint8_t> bool_key, uint64_t hash_key, IdType id) {

        std::pair<uint64_t, IdType> element(hash_key, id);
        std::vector<std::pair<uint64_t, IdType> > element_list(1, element);
        dict[bool_key] = element_list;
    }

    std::vector<std::pair<uint64_t, IdType> > get(uint32_t key) {
        std::vector<uint8_t> bool_key = actual_key(key);

        if (dict.count(bool_key) > 0)
            return dict[bool_key];
        else {
            std::pair<uint64_t, IdType> element(0, *new IdType());
            std::vector<std::pair<uint64_t, IdType> > element_list(1, element);
            return element_list;
        }
    }

    bool exist(uint32_t key) {
        std::vector<uint8_t> bool_key = actual_key(key);

        if (dict.count(bool_key) > 0)
            return true;
        else
            return false;
    }

    void clear() { dict.clear(); }

    void merge(FastDict<IdType>& source) {

        std::pair<std::vector<uint8_t>, std::vector<std::pair<uint64_t, IdType> > > me;
        BOOST_FOREACH(me, source.dict) {
            std::pair<uint64_t, IdType> element;
            BOOST_FOREACH(element, me.second) {
                dict[me.first].insert(dict[me.first].end(), element);
            }
        }

    }

    uint32_t size() { return dict.size(); }

    void append(uint32_t key, uint64_t hash_key, IdType id) {
        std::vector<uint8_t> bool_key = actual_key(key);

        std::pair<uint64_t, IdType> element(hash_key, id);
        dict[bool_key].insert(dict[bool_key].end(), element);
    }

    std::vector<uint32_t> keys() {
        std::pair<std::vector<uint8_t>, std::vector<std::pair<uint64_t, IdType> > > me;
        std::vector<uint32_t> keys;
        BOOST_FOREACH(me, dict) {
            keys.push_back(python_key(me.first));
        }

        return keys;
    }

    void set_keydimensions(boost::python::list& dimensions) {
        for (int i = 0; i < len(dimensions); ++i) {
            key_dimensions.insert(key_dimensions.end(), boost::python::extract<int>(dimensions[i]));
        }
    }

    void get_keydimensions(boost::python::list& dimensions) {
        BOOST_FOREACH(uint32_t dim, key_dimensions) {
            dimensions.append(dim);
        }
    }

    // because we allow python program to retrieve elements in this dictionary by int (or long?) key
    // we should generate actual key from uint32_t (or uint64_t) key of python
    std::vector<uint8_t> actual_key(uint32_t python_key) {
        std::vector<uint8_t> key;
        uint8_t current_bits = 0;
        uint32_t base = 0xFF000000;


        // for test
        // std::cout << "In actual_key() python key: " << int(python_key) << "\n";

        // for efficiency consideration, index_key_dimension should be divisible by 8

        // int times = index_key_dimension / 8;
        for (int i = 0; i < 4; ++i) {
            key.insert(key.end(), (uint8_t)((python_key & base) >> (8 * (4 - i - 1))));
            base = base >> 8;
        }

        // for test
        /*
        BOOST_FOREACH(uint8_t k, key) {
            std::cout << " " << int(k);
        }
        std::cout << "\n";
        */

        /*
        for (int i = 0; i < index_key_dimension; ++i) {

            if ((python_key & 0x01) == 1)
                current_bits = current_bits + (0x01 << (i % 8));

            python_key = python_key >> 1;

            if ((i + 1) % 8 == 0) {
                key.insert(key.begin(), current_bits);
                current_bits = 0;
            }
        }
        */           
        return key;
    }

    uint32_t python_key(std::vector<uint8_t> key) {
        uint32_t p_key = 0;
        uint8_t cur_bit = 0;

        for (std::vector<uint8_t>::reverse_iterator rit = key.rbegin(); rit != key.rend(); ++rit) {
            uint8_t bits = *rit;

            for (uint8_t i = 0; i < 8; i++) {

                if ((bits & 0x01) == 1)
                    p_key = p_key + (0x01 << cur_bit);

                cur_bit++;

                bits = bits >> 1;
            }
        }

        // for test
        // std::cout << "In python_key() python key: " << long(p_key) << "\n";

        return p_key;
    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & dict;
        ar & key_dimensions;
        ar & index_key_dimension;
    }

    // internally, we use a vector of uint8_t values as key of indexing (dictionary)

    std::map<std::vector<uint8_t>, std::vector<std::pair<uint64_t, IdType> > > dict;
    std::vector<uint32_t> key_dimensions;

    uint8_t index_key_dimension;
};


template <class IdType>
bool sort_func(std::pair<uint64_t, IdType> first, std::pair<uint64_t, IdType> second) {
    return (first.first < second.first);
}
 
template <class BitCountType, class IdType>
class FastCompressDict: public FastDict<IdType> {

public:
    typedef FastDict<IdType> super;

    FastCompressDict(uint8_t k_dim) : FastDict<IdType>(k_dim) {}

    friend class boost::serialization::access;

    void merge(FastCompressDict<BitCountType, IdType>& source) {

        std::pair<std::vector<uint8_t>, std::vector<std::pair<uint64_t, IdType> > > me;
        BOOST_FOREACH(me, source.dict) {
            std::pair<uint64_t, IdType> element;
            BOOST_FOREACH(element, me.second) {
                super::dict[me.first].insert(super::dict[me.first].end(), element);
            }
        }

    }

    void go_index() {

        /*
        BOOST_FOREACH(me, super::dict) {
            // sort binart codes in each bucket
            // std::vector<std::pair<uint64_t, IdType> > vec = me.second;
            std::sort(me.second.begin(), me.second.end(), sort_func<IdType>);

            
            // for test
            std::pair<uint64_t, IdType> element;

            BOOST_FOREACH(element, me.second) {
                std::cout << ' ' << element.first;
            }
            std::cout << '\n';
        }
        */

        std::pair<std::vector<uint8_t>, std::vector<std::pair<uint64_t, IdType> > > me;

        BOOST_FOREACH(me, super::dict) {
            // for binary codes in each bucket

            // sort binart codes in each bucket
            std::sort(me.second.begin(), me.second.end(), sort_func<IdType>);

            // for test
            /*
            std::pair<uint64_t, IdType> element_test;

            std::cout << "bucket: ";

            BOOST_FOREACH(element_test, me.second) {
                std::cout << ' ' << element_test.first;
            }
            std::cout << '\n';
            */
 
            // generate column-based representation for binary codes in each bucket

            std::vector<std::vector<uint8_t> > columns(64, *new std::vector<uint8_t>());
            std::pair<uint64_t, IdType> element;
            std::vector<IdType> id_vector;

            BOOST_FOREACH(element, me.second) {
                uint64_t binary_code = element.first;
 
                // test
                // std::cout << "code: " << uint64_t(binary_code) << ' ' << int(element.second) << "\n";
 

                for (uint8_t i = 0; i < 64; i++) {
                    if ((binary_code & 0x01) == 1) {
                        columns[i].push_back(1);    
                    } else {
                        columns[i].push_back(0);
                    }
                    binary_code = binary_code >> 1; 
                }
    
                id_vector.push_back(element.second);
            }
 
            // for test
            /*
            uint8_t index= 0;
            BOOST_FOREACH(std::vector<uint8_t> column, columns) {
                std::cout << int(index++) << ": ";
                BOOST_FOREACH(uint8_t bit, column) {
                    std::cout << int(bit) << ' ';
                } 
                std::cout << '\n';
            }
            index = 0;
            */

            //  compress data
            std::vector<std::vector<BitCountType> > compress_data(64, *new std::vector<BitCountType>());
            uint8_t column_index = 0;
            BOOST_FOREACH(std::vector<uint8_t> column, columns) {
                //  scan each column to compress the data
                uint8_t prev_repeat_bit = 0;
                BitCountType repeat_count = 0;
                BOOST_FOREACH(uint8_t bit, column) {
                    if (bit == prev_repeat_bit) {
                        repeat_count++;
                    } else {
                        compress_data[column_index].push_back(repeat_count);        
                        prev_repeat_bit = bit;
                        repeat_count = 1;
                    }
                }
                compress_data[column_index].push_back(repeat_count);
                column_index++;
            }

            
            // test
            /*
            BOOST_FOREACH(std::vector<uint8_t> column, compress_data) {
                std::cout << int(index++) << ": ";
                BOOST_FOREACH(uint8_t bit, column) {
                    std::cout << int(bit) << ' ';
                } 
                std::cout << '\n';
            }
            */
            
            std::pair<std::vector<std::vector<BitCountType> >, std::vector<IdType> > pair(compress_data, id_vector);
            column_dict[me.first] = pair;
            //super::set_with_bool_key(me.first, 0x00, *new IdType());
            //super::dict.erase(me.first);

        }
        super::dict.clear();
    }


    // test for buffer
    /*
    std::pair<std::vector<PyObject>, std::vector<IdType> >  get_cols_as_buffer(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        std::vector<PyObject> buffers(0);

        if (column_dict.count(bool_key) > 0) {

                std::vector<BitCountType> column;
                BOOST_FOREACH(column, column_dict[bool_key].first) {               
                    void* data = (void*)(column.data());
                    Py_ssize_t size = column.size();
                    buffers.insert(buffers.end(), *PyBuffer_FromMemory (data, size));
                }
        }

        std::pair<std::vector<PyObject>, std::vector<IdType> > apair(buffers, column_dict[bool_key].second);

        return apair;
    }
    */

    // at begining, we use std vector to provide raw data (via .data()) to python buffer
    // however, the raw data is broken and we can not obtain correct data at python side
    // but using plain array is OK.
    // since dynamically create array at every time python sends query is too slow
    // so we add this method to be called before any querying of compressed dict.
    void init_runtime_dict() {

        std::pair<std::vector<uint8_t>, std::pair<std::vector<std::vector<BitCountType> >, std::vector<IdType> > > me;

        BOOST_FOREACH(me, column_dict) {
            std::vector<uint64_t*> columns;
            std::vector<uint32_t> columns_length;

            std::vector<BitCountType> column;
            BOOST_FOREACH(column, me.second.first) {               
                uint64_t* column_as_array =  new uint64_t[column.size()];

                int ele_index = 0;
                BOOST_FOREACH(BitCountType ele, column) {
                    column_as_array[ele_index++] = (uint64_t)ele;
                }
                columns.insert(columns.end(), column_as_array);
                columns_length.insert(columns_length.end(), column.size());
            }

            std::pair<std::vector<uint64_t*>, std::vector<IdType> > nested_pair(columns, me.second.second);
            std::pair<std::vector<uint32_t>, std::pair<std::vector<uint64_t*>, std::vector<IdType> > > pair(columns_length, nested_pair);
            runtime_dict[me.first] = pair;
        }
        column_dict.clear();
    }
 
    std::vector<PyObject*> get_cols_as_buffer(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        std::vector<PyObject*> buffers(0);

        if (runtime_dict.count(bool_key) > 0) {

                uint64_t* column;
                int column_index = 0;
                BOOST_FOREACH(column, runtime_dict[bool_key].second.first) {               

                    /*
                    uint64_t* column_as_array =  new uint64_t[column.size()];

                    int ele_index = 0;
                    BOOST_FOREACH(BitCountType ele, column) {
                        column_as_array[ele_index++] = (uint64_t)ele;
                    }
                    */

                    PyObject* buffer_obj = PyBuffer_FromMemory ((void*)column, runtime_dict[bool_key].first[column_index++] * 8);
                    boost::python::incref(buffer_obj);

                    buffers.insert(buffers.end(), buffer_obj);
                }
        }

        return buffers;
    }
 
    std::vector<IdType> get_image_ids(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        if (runtime_dict.count(bool_key) > 0)
            return runtime_dict[bool_key].second.second;
        else {
            std::vector<IdType> id_vector(0);
            return id_vector;
        }
    }
 
    // get raw compressed data
    // called before init_runtime_dict() since initialization will clear column_dict
    std::pair<std::vector<std::vector<BitCountType> >, std::vector<IdType> > get_cols(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        if (column_dict.count(bool_key) > 0)
            return column_dict[bool_key];
        else {
            //std::vector<std::vector<uint8_t> > columns(1, *new std::vector<uint8_t>(1, 0));
            //std::vector<IdType> id_vector(1, *new IdType());
            std::vector<std::vector<BitCountType> > columns(0);
            std::vector<IdType> id_vector(0);
            std::pair<std::vector<std::vector<BitCountType> >, std::vector<IdType> > empty_pair(columns, id_vector);
            return empty_pair;
        }
    }

    // cpu-based uncompression algorithm
    std::pair<std::vector<uint64_t>, std::vector<IdType> > get_binary_codes(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);
        std::vector<uint64_t> binary_codes;

        if (column_dict.count(bool_key) > 0) {
            uint32_t current_binary_code_num = 0;
            for (int binary_code_count = 0; binary_code_count < column_dict[bool_key].second.size(); binary_code_count++) {
                std::vector<BitCountType> column; 
                uint64_t binary_code = 0x00;
                uint16_t column_index = 0;
                BOOST_FOREACH(column, column_dict[bool_key].first) {
                    uint32_t count_for_bits = 0;
                    uint8_t bit_type = 0x00;
                    BOOST_FOREACH(BitCountType bit_counts, column) {
                        count_for_bits += bit_counts;

                        // for test
                        // std::cout << "count_for_bits: " << count_for_bits << "\n";
                        // std::cout << "current_binary_code_num: " << current_binary_code_num << "\n";

                        if (count_for_bits > current_binary_code_num) {
                            if (bit_type == 1)
                                binary_code = binary_code | ((uint64_t)1 << column_index);

                            // for test
                            // std::cout << "column_index: " << column_index << "\n";
                            // std::cout << "inter. binary_code: " <<  binary_code  << "\n";
 
                            column_index++;
                            break;
                        }
                        bit_type = bit_type ^ 0x01;
                    } 
                }
                // for test
                // std::cout << "binary_code: " << long(binary_code) << "\n";

                current_binary_code_num++;
                binary_codes.push_back(binary_code);
            }
            std::pair<std::vector<uint64_t>, std::vector<IdType> > apair(binary_codes, column_dict[bool_key].second);
            return apair;
        }
        else {
            std::vector<uint64_t> binary_codes(0);
            std::vector<IdType> id_vector(0);
            std::pair<std::vector<uint64_t>, std::vector<IdType> > empty_pair(binary_codes, id_vector);
            return empty_pair;
        }
 
    }



    std::map<std::vector<uint8_t>, std::pair<std::vector<std::vector<BitCountType> >, std::vector<IdType> > > column_dict;
    std::map<std::vector<uint8_t>, std::pair<std::vector<uint32_t>, std::pair<std::vector<uint64_t*>, std::vector<IdType> > > > runtime_dict;

};

template <class IdType>
void save(char* filename, FastDict<IdType> dict) {
   std::ofstream ofs(filename);

   boost::archive::text_oarchive oa(ofs);
   oa << dict.dict;
   oa << dict.key_dimensions;
   oa << dict.index_key_dimension;
}

template <class IdType>
void load(char* filename, FastDict<IdType>& dict) {
    std::ifstream ifs(filename);

    boost::archive::text_iarchive ia(ifs);
    ia >> dict.dict;
    ia >> dict.key_dimensions;
    ia >> dict.index_key_dimension;

}
 
template <class BitCountType, class IdType>
void save_compress(char* filename, FastCompressDict<BitCountType, IdType> dict) {
   std::ofstream ofs(filename);

   boost::archive::text_oarchive oa(ofs);
   oa << dict.dict;
   oa << dict.key_dimensions;
   oa << dict.index_key_dimension;
   oa << dict.column_dict;
}

template <class BitCountType, class IdType>
void load_compress(char* filename, FastCompressDict<BitCountType, IdType>& dict) {
    std::ifstream ifs(filename);

    boost::archive::text_iarchive ia(ifs);
    ia >> dict.dict;
    ia >> dict.key_dimensions;
    ia >> dict.index_key_dimension;
    ia >> dict.column_dict;

}
 
using namespace boost::python;

BOOST_PYTHON_MODULE(fastdict)
{

    to_python_converter<
        PyObject,
        pyobject_to_python>();
 
    to_python_converter<
        std::vector<PyObject*>,
        pyobject_vector_to_python>();

    class_<FastDict<std::string> >("FastDict", init<uint8_t>())
        .def("get", &FastDict<std::string>::get)
        .def("set", &FastDict<std::string>::set)
        .def("append", &FastDict<std::string>::append)
        .def("size", &FastDict<std::string>::size)
        .def("keys", &FastDict<std::string>::keys)
        .def("set_keydimensions", &FastDict<std::string>::set_keydimensions)
        .def("get_keydimensions", &FastDict<std::string>::get_keydimensions)
        .def("exist", &FastDict<std::string>::exist)
        .def("clear", &FastDict<std::string>::clear)
        .def("merge", &FastDict<std::string>::merge)
    ;

    class_<std::vector<std::pair<uint64_t, std::string> > >("PairVec")
        .def(vector_indexing_suite<std::vector<std::pair<uint64_t, std::string> > >())
    ;

    class_<std::pair<uint64_t, std::string> >("HashPair")
        .def_readwrite("first", &std::pair<uint64_t, std::string>::first)
        .def_readwrite("second", &std::pair<uint64_t, std::string>::second)
    ;

    class_<std::vector<uint32_t> >("ShorKeyVec")
        .def(vector_indexing_suite<std::vector<uint32_t> >())
    ;
 

    def("save", save<std::string>);
    def("load", load<std::string>);

    class_<FastDict<uint32_t> >("FastIntDict", init<uint8_t>())
        .def("get", &FastDict<uint32_t>::get)
        .def("set", &FastDict<uint32_t>::set)
        .def("append", &FastDict<uint32_t>::append)
        .def("size", &FastDict<uint32_t>::size)
        .def("keys", &FastDict<uint32_t>::keys)
        .def("set_keydimensions", &FastDict<uint32_t>::set_keydimensions)
        .def("get_keydimensions", &FastDict<uint32_t>::get_keydimensions)
        .def("exist", &FastDict<uint32_t>::exist)
        .def("clear", &FastDict<uint32_t>::clear)
        .def("merge", &FastDict<uint32_t>::merge)
    ;

    class_<std::vector<std::pair<uint64_t, uint32_t> > >("PairIntVec")
        .def(vector_indexing_suite<std::vector<std::pair<uint64_t, uint32_t> > >())
    ;                                                           

    class_<std::pair<uint64_t, uint32_t> >("HashIntPair")
        .def_readwrite("first", &std::pair<uint64_t, uint32_t>::first)
        .def_readwrite("second", &std::pair<uint64_t, uint32_t>::second)
    ;

    def("save_int", save<uint32_t>);
    def("load_int", load<uint32_t>);

    class_<FastCompressDict<uint8_t, uint32_t> >("FastCompressIntDict", init<uint8_t>())
        .def("get", &FastCompressDict<uint8_t, uint32_t>::get)
        .def("set", &FastCompressDict<uint8_t, uint32_t>::set)
        .def("append", &FastCompressDict<uint8_t, uint32_t>::append)
        .def("size", &FastCompressDict<uint8_t, uint32_t>::size)
        .def("keys", &FastCompressDict<uint8_t, uint32_t>::keys)
        .def("set_keydimensions", &FastCompressDict<uint8_t, uint32_t>::set_keydimensions)
        .def("get_keydimensions", &FastCompressDict<uint8_t, uint32_t>::get_keydimensions)
        .def("exist", &FastCompressDict<uint8_t, uint32_t>::exist)
        .def("clear", &FastCompressDict<uint8_t, uint32_t>::clear)
        .def("merge", &FastCompressDict<uint8_t, uint32_t>::merge)
        .def("go_index", &FastCompressDict<uint8_t, uint32_t>::go_index)
        .def("get_cols", &FastCompressDict<uint8_t, uint32_t>::get_cols)
        .def("get_binary_codes", &FastCompressDict<uint8_t, uint32_t>::get_binary_codes)
        .def("get_cols_as_buffer", &FastCompressDict<uint8_t, uint32_t>::get_cols_as_buffer)
        .def("get_image_ids", &FastCompressDict<uint8_t, uint32_t>::get_image_ids)
        .def("init_runtime_dict", &FastCompressDict<uint8_t, uint32_t>::init_runtime_dict)
    ;

    class_<std::vector<std::vector<uint8_t> > >("CompressedUInt8ColumnIntVec")
        .def(vector_indexing_suite<std::vector<std::vector<uint8_t> > >())
    ;

    class_<std::vector<uint8_t> >("UInt8BitCountVec")
        .def(vector_indexing_suite<std::vector<uint8_t> >())
    ;
 
    class_<std::vector<uint64_t> >("BinaryCodesVec")
        .def(vector_indexing_suite<std::vector<uint64_t> >())
    ;

    /*
    class_<std::pair<std::vector<PyObject>, std::vector<uint32_t> > >("CompressedBufferColumnsIdsIntPair")
        .def_readwrite("first", &std::pair<std::vector<PyObject>, std::vector<uint32_t> >::first)
        .def_readwrite("second", &std::pair<std::vector<PyObject>, std::vector<uint32_t> >::second)
    ;

    class_<std::vector<PyObject> >("BufferVec")
        .def(vector_indexing_suite<std::vector<PyObject> >())
    ;
    */
 
    class_<std::pair<std::vector<std::vector<uint8_t> >, std::vector<uint32_t> > >("CompressedColumnsIdsIntPair")
        .def_readwrite("first", &std::pair<std::vector<std::vector<uint8_t> >, std::vector<uint32_t> >::first)
        .def_readwrite("second", &std::pair<std::vector<std::vector<uint8_t> >, std::vector<uint32_t> >::second)
    ;
 
    class_<std::pair<std::vector<uint64_t>, std::vector<uint32_t> > >("BinaryCodessIdsIntPair")
        .def_readwrite("first", &std::pair<std::vector<uint64_t>, std::vector<uint32_t> >::first)
        .def_readwrite("second", &std::pair<std::vector<uint64_t>, std::vector<uint32_t> >::second)
    ;
 

    def("save_compress_int", save_compress<uint8_t, uint32_t>);
    def("load_compress_int", load_compress<uint8_t, uint32_t>);
 
    // CompressDict for storing bit counts in uint32_t type

    class_<FastCompressDict<uint32_t, uint32_t> >("FastCompressUInt32IntDict", init<uint8_t>())
        .def("get", &FastCompressDict<uint32_t, uint32_t>::get)
        .def("set", &FastCompressDict<uint32_t, uint32_t>::set)
        .def("append", &FastCompressDict<uint32_t, uint32_t>::append)
        .def("size", &FastCompressDict<uint32_t, uint32_t>::size)
        .def("keys", &FastCompressDict<uint32_t, uint32_t>::keys)
        .def("set_keydimensions", &FastCompressDict<uint32_t, uint32_t>::set_keydimensions)
        .def("get_keydimensions", &FastCompressDict<uint32_t, uint32_t>::get_keydimensions)
        .def("exist", &FastCompressDict<uint32_t, uint32_t>::exist)
        .def("clear", &FastCompressDict<uint32_t, uint32_t>::clear)
        .def("merge", &FastCompressDict<uint32_t, uint32_t>::merge)
        .def("go_index", &FastCompressDict<uint32_t, uint32_t>::go_index)
        .def("get_cols", &FastCompressDict<uint32_t, uint32_t>::get_cols)
        .def("get_binary_codes", &FastCompressDict<uint32_t, uint32_t>::get_binary_codes)
        .def("get_cols_as_buffer", &FastCompressDict<uint32_t, uint32_t>::get_cols_as_buffer)
        .def("get_image_ids", &FastCompressDict<uint32_t, uint32_t>::get_image_ids)
        .def("init_runtime_dict", &FastCompressDict<uint32_t, uint32_t>::init_runtime_dict)
    ;

    class_<std::vector<std::vector<uint32_t> > >("CompressedUInt32ColumnIntVec")
        .def(vector_indexing_suite<std::vector<std::vector<uint32_t> > >())
    ;

    class_<std::vector<uint32_t> >("UInt32BitCountVec")
        .def(vector_indexing_suite<std::vector<uint32_t> >())
    ;
 
    class_<std::pair<std::vector<std::vector<uint32_t> >, std::vector<uint32_t> > >("CompressedUInt32ColumnsIdsIntPair")
        .def_readwrite("first", &std::pair<std::vector<std::vector<uint32_t> >, std::vector<uint32_t> >::first)
        .def_readwrite("second", &std::pair<std::vector<std::vector<uint32_t> >, std::vector<uint32_t> >::second)
    ;
 
    def("save_compress_uint32_int", save_compress<uint32_t, uint32_t>);
    def("load_compress_uint32_int", load_compress<uint32_t, uint32_t>);
 
}


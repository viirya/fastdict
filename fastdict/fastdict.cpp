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

//#include <boost/numpy.hpp>


// base64 encoding table
static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

 
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
 
struct vector_pyobject_vector_to_python
{
    static PyObject* convert(std::vector<std::vector<PyObject*> > const& array)
    { 
        boost::python::list pylist;

        std::vector<PyObject*> vector;
        BOOST_FOREACH(vector, array) {
            pylist.append(vector);
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

    std::vector<std::pair<uint64_t, IdType> > mget(boost::python::list& keys) {
        std::vector<std::pair<uint64_t, IdType> > return_keys(0);
        for (int i = 0; i < len(keys); i++) {
            std::vector<uint8_t> bool_key = actual_key(boost::python::extract<uint32_t>(keys[i]));
            
            if (dict.count(bool_key) > 0) {
                std::pair<uint64_t, IdType> element;
                BOOST_FOREACH(element, dict[bool_key]) {
                    return_keys.insert(return_keys.end(), element);
                }
            }
        }
        return return_keys;
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
 
    void batch_append(boost::python::list& keys, boost::python::list& hash_keys, boost::python::list& ids) {
        std::vector<std::vector<uint8_t> > bool_keys(len(keys));        
        for (int i = 0; i < len(keys); i++) {            
            bool_keys[i] = actual_key(boost::python::extract<uint32_t>(keys[i]));            
            dict[bool_keys[i]].reserve(len(keys));        
        }        

        for (int i = 0; i < len(keys); i++) {            
            std::vector<uint8_t> bool_key = bool_keys[i];

            std::pair<uint64_t, IdType> element(boost::python::extract<uint64_t>(hash_keys[i]), boost::python::extract<IdType>(ids[i]));
            dict[bool_key].insert(dict[bool_key].end(), element);        
        }
    }
 
    void batch_iter_append(boost::python::list& keys, boost::python::list& hash_keys, boost::python::list& ids) {

        std::map<std::vector<uint8_t>, std::vector<std::pair<uint64_t, IdType> > > tmp_dict;

        for (int i = 0; i < len(keys); i++) {
            std::vector<uint8_t> bool_key = actual_key(boost::python::extract<uint32_t>(keys[i]));

            std::pair<uint64_t, IdType> element(boost::python::extract<uint64_t>(hash_keys[i]), boost::python::extract<IdType>(ids[i]));
            tmp_dict[bool_key].insert(tmp_dict[bool_key].end(), element);
        }

        std::pair<std::vector<uint8_t>, std::vector<std::pair<uint64_t, IdType> > > me;

        BOOST_FOREACH(me, tmp_dict) {
            dict[me.first].insert(dict[me.first].end(), tmp_dict[me.first].begin(), tmp_dict[me.first].end());
        }

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
        for (int i = 0; i < 4; ++i) {
            current_bits = (uint8_t)((python_key & base) >> (8 * (4 - i - 1)));
            if (index_key_dimension > (8 * (4 - i - 1)))
                key.insert(key.end(), current_bits);
            base = base >> 8;
        }

        // for test
        /*
        std::cout << "key: ";
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

    FastCompressDict(uint8_t k_dim) : FastDict<IdType>(k_dim) { dict_status = -1; }

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

                // make the length of compressed column even
                // it is for concatenating corresponding columns of different buckets later in python code
                if (compress_data[column_index].size() % 2 != 0)
                    compress_data[column_index].push_back(0);
                
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

        dict_status = 0;
    }

    // convert column_dict to VLQ base64 format
    void to_VLQ_base64_dict() {

        std::pair<std::vector<uint8_t>, std::pair<std::vector<std::vector<BitCountType> >, std::vector<IdType> > > me;

        BOOST_FOREACH(me, column_dict) {
            std::vector<std::string> columns;

            std::vector<BitCountType> column;
            BOOST_FOREACH(column, me.second.first) {               
                std::string column_as_VLQ_base64 = "";

                BOOST_FOREACH(BitCountType ele, column) {
                    column_as_VLQ_base64 += base64VLQ_encode(ele);
                }
                columns.insert(columns.end(), column_as_VLQ_base64);
            }

            std::pair<std::vector<std::string>, std::vector<IdType> > columns_pair(columns, me.second.second);
            column_vlq_dict[me.first] = columns_pair;
        }
        column_dict.clear();

        dict_status = 1;
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
            std::vector<BitCountType*> columns;
            std::vector<uint32_t> columns_length;

            std::vector<BitCountType> column;
            BOOST_FOREACH(column, me.second.first) {               
                BitCountType* column_as_array =  new BitCountType[column.size()];

                int ele_index = 0;
                BOOST_FOREACH(BitCountType ele, column) {
                    column_as_array[ele_index++] = ele;
                }
                columns.insert(columns.end(), column_as_array);
                columns_length.insert(columns_length.end(), column.size());
            }

            std::pair<std::vector<BitCountType*>, std::vector<IdType> > nested_pair(columns, me.second.second);
            std::pair<std::vector<uint32_t>, std::pair<std::vector<BitCountType*>, std::vector<IdType> > > pair(columns_length, nested_pair);
            runtime_dict[me.first] = pair;
        }
        column_dict.clear();

        dict_status = 2;
    }
 
    // initiate runtime dict for VLQ base64 column dict
    void init_runtime_VLQ_base64_dict() {

        std::pair<std::vector<uint8_t>, std::pair<std::vector<std::string>, std::vector<IdType> > > me;

        BOOST_FOREACH(me, column_vlq_dict) {
            std::vector<char*> columns;
            std::vector<uint32_t> columns_length;

            BOOST_FOREACH(std::string column, me.second.first) {               
                char* column_as_array = new char[column.size()];

                for (int str_index = 0; str_index < column.length(); str_index++)
                    column_as_array[str_index] = (char)column[str_index];

                columns.insert(columns.end(), column_as_array);
                columns_length.insert(columns_length.end(), column.size());
            }

            std::pair<std::vector<char*>, std::vector<IdType> > nested_pair(columns, me.second.second);
            std::pair<std::vector<uint32_t>, std::pair<std::vector<char*>, std::vector<IdType> > > pair(columns_length, nested_pair);
            runtime_vlq_dict[me.first] = pair;
        }
        column_vlq_dict.clear();

        dict_status = 3;
    }

    // for non VQL base64 runtime dict 
    std::vector<PyObject*> get_cols_as_buffer(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        std::vector<PyObject*> buffers(0);

        if (runtime_dict.count(bool_key) > 0) {

                BitCountType* column;
                int column_index = 0;
                BOOST_FOREACH(column, runtime_dict[bool_key].second.first) {               

                    /*
                    uint64_t* column_as_array =  new uint64_t[column.size()];

                    int ele_index = 0;
                    BOOST_FOREACH(BitCountType ele, column) {
                        column_as_array[ele_index++] = (uint64_t)ele;
                    }
                    */

                    PyObject* buffer_obj = PyBuffer_FromMemory ((void*)column, runtime_dict[bool_key].first[column_index++] * sizeof(BitCountType));
                    boost::python::incref(buffer_obj);

                    buffers.insert(buffers.end(), buffer_obj);
                }
        }

        return buffers;
    }
 
    // for non VQL base64 runtime dict 
    std::vector<std::vector<PyObject*> > mget_cols_as_buffer(boost::python::list& keys) {

        std::vector<std::vector<PyObject*> > return_vector(0);

        for (int i = 0; i < len(keys); i++) {
            return_vector.insert(return_vector.end(), get_cols_as_buffer(boost::python::extract<uint32_t>(keys[i])));
        }
        return return_vector;
    }

 
    // for VLQ base64 runtime dict
    std::vector<PyObject*> get_VLQ_base64_cols_as_buffer(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        std::vector<PyObject*> buffers(0);

        if (runtime_vlq_dict.count(bool_key) > 0) {

                char* column;
                int column_index = 0;
                BOOST_FOREACH(column, runtime_vlq_dict[bool_key].second.first) {               

                    PyObject* buffer_obj = PyBuffer_FromMemory ((void*)column, runtime_vlq_dict[bool_key].first[column_index++]);
                    boost::python::incref(buffer_obj);

                    buffers.insert(buffers.end(), buffer_obj);
                }
        }

        return buffers;
    }

    // for VLQ base64 runtime dict
    std::vector<std::vector<PyObject*> > mget_VLQ_base64_cols_as_buffer(boost::python::list& keys) {

        std::vector<std::vector<PyObject*> > return_vector(0);

        for (int i = 0; i < len(keys); i++) {
            return_vector.insert(return_vector.end(), get_VLQ_base64_cols_as_buffer(boost::python::extract<uint32_t>(keys[i])));
        }
        return return_vector;

    }
 
    // called after init runtime dict
    std::vector<IdType> get_image_ids(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        if (runtime_dict.count(bool_key) > 0)
            return runtime_dict[bool_key].second.second;
        else {
            std::vector<IdType> id_vector(0);
            return id_vector;
        }
    }

    // called after init runtime dict 
    std::vector<IdType> mget_image_ids(boost::python::list& keys) {
        std::vector<IdType> image_ids(0);

        for (int i = 0; i < len(keys); i++) {
            BOOST_FOREACH(IdType id, get_image_ids(boost::python::extract<uint32_t>(keys[i]))) {
                image_ids.insert(image_ids.end(), id);
            }
        }
        return image_ids;
    }
 
    // for VLQ base64 runtime dict
    // called after init VLQ base64 runtime dict
    std::vector<IdType> get_VLQ_base64_image_ids(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        if (runtime_vlq_dict.count(bool_key) > 0)
            return runtime_vlq_dict[bool_key].second.second;
        else {
            std::vector<IdType> id_vector(0);
            return id_vector;
        }
    }

    // for VLQ base64 runtime dict
    std::vector<IdType> mget_VLQ_base64_image_ids(boost::python::list& keys) {
        std::vector<IdType> image_ids(0);

        for (int i = 0; i < len(keys); i++) {
            BOOST_FOREACH(IdType id, get_VLQ_base64_image_ids(boost::python::extract<uint32_t>(keys[i]))) {
                image_ids.insert(image_ids.end(), id);
            }
        }
        return image_ids;
    }
 
    // called before init runtime dict 
    std::vector<IdType> get_image_ids_before_runtime_init(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        if (column_dict.count(bool_key) > 0)
            return column_dict[bool_key].second;
        else {
            std::vector<IdType> id_vector(0);
            return id_vector;
        }
    }
 
    // called before init runtime dict 
    std::vector<IdType> mget_image_ids_before_runtime_init(boost::python::list& keys) {
        std::vector<IdType> image_ids(0);

        for (int i = 0; i < len(keys); i++) {
            BOOST_FOREACH(IdType id, get_image_ids_before_runtime_init(boost::python::extract<uint32_t>(keys[i]))) {
                image_ids.insert(image_ids.end(), id);
            }
        }
        return image_ids;
    }
 
    // called before init runtime dict 
    // for VLQ base64 column dict
    std::vector<IdType> get_VLQ_base64_image_ids_before_runtime_init(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        if (column_vlq_dict.count(bool_key) > 0)
            return column_vlq_dict[bool_key].second;
        else {
            std::vector<IdType> id_vector(0);
            return id_vector;
        }
    }
 
    // called before init runtime dict 
    // for VLQ base64 column dict
    std::vector<IdType> mget_VLQ_base64_image_ids_before_runtime_init(boost::python::list& keys) {
        std::vector<IdType> image_ids(0);

        for (int i = 0; i < len(keys); i++) {
            BOOST_FOREACH(IdType id, get_VLQ_base64_image_ids_before_runtime_init(boost::python::extract<uint32_t>(keys[i]))) {
                image_ids.insert(image_ids.end(), id);
            }
        }
        return image_ids;
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

    // for VLQ base64     
    std::pair<std::vector<std::string>, std::vector<IdType> > get_VLQ_base64_cols(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);

        if (column_vlq_dict.count(bool_key) > 0)
            return column_vlq_dict[bool_key];
        else {
            //std::vector<std::vector<uint8_t> > columns(1, *new std::vector<uint8_t>(1, 0));
            //std::vector<IdType> id_vector(1, *new IdType());
            std::vector<std::string> columns(0);
            std::vector<IdType> id_vector(0);
            std::pair<std::vector<std::string>, std::vector<IdType> > empty_pair(columns, id_vector);
            return empty_pair;
        }
    }
 
    // cpu-based uncompression algorithm
    // only workable before init runtime dict
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
 
    // cpu-based uncompression algorithm
    // only workable before init runtime dict
    std::pair<std::vector<uint64_t>, std::vector<IdType> > mget_binary_codes(boost::python::list& keys) {
        std::vector<uint64_t> binary_codes(0);
        std::vector<IdType> id_vector(0);
        std::pair<std::vector<uint64_t>, std::vector<IdType> > return_pair(binary_codes, id_vector);

        for (int i = 0; i < len(keys); i++) {
            std::pair<std::vector<uint64_t>, std::vector<IdType> > partial_binary_codes = get_binary_codes(boost::python::extract<uint32_t>(keys[i]));

            BOOST_FOREACH(uint64_t binary_code, partial_binary_codes.first) {
                return_pair.first.insert(return_pair.first.end(), binary_code);
            }
            BOOST_FOREACH(IdType binary_code_id, partial_binary_codes.second) {
                return_pair.second.insert(return_pair.second.end(), binary_code_id);
            }
        }
        return return_pair;
    }
 
    // cpu-based uncompression algorithm for VLQ base64 compressed dict
    // only workable before init VLQ base64 runtime dict
    std::pair<std::vector<uint64_t>, std::vector<IdType> > get_VLQ_base64_binary_codes(uint32_t key) {
        std::vector<uint8_t> bool_key = super::actual_key(key);
        std::vector<uint64_t> binary_codes;

        if (column_vlq_dict.count(bool_key) > 0) {
            uint32_t current_binary_code_num = 0;
            for (int binary_code_count = 0; binary_code_count < column_vlq_dict[bool_key].second.size(); binary_code_count++) {
                uint64_t binary_code = 0x00;
                uint16_t column_index = 0;
                BOOST_FOREACH(std::string column, column_vlq_dict[bool_key].first) {
                    uint32_t count_for_bits = 0;
                    uint8_t bit_type = 0x00;
                    uint32_t VLQ_base64_string_offset = 0;

                    std::pair<BitCountType, uint32_t> decode_pair;

                    while (VLQ_base64_string_offset < column.size()) {
                        decode_pair = incre_base64VLQ_decode(column, VLQ_base64_string_offset);
                        BitCountType bit_counts = decode_pair.first;
                        VLQ_base64_string_offset = decode_pair.second;

                        count_for_bits += bit_counts;

                        if (count_for_bits > current_binary_code_num) {
                            if (bit_type == 1)
                                binary_code = binary_code | ((uint64_t)1 << column_index);

                            column_index++;
                            break;
                        }
                        bit_type = bit_type ^ 0x01;
                    } 
                }

                current_binary_code_num++;
                binary_codes.push_back(binary_code);
            }
            std::pair<std::vector<uint64_t>, std::vector<IdType> > apair(binary_codes, column_vlq_dict[bool_key].second);
            return apair;
        }
        else {
            std::vector<uint64_t> binary_codes(0);
            std::vector<IdType> id_vector(0);
            std::pair<std::vector<uint64_t>, std::vector<IdType> > empty_pair(binary_codes, id_vector);
            return empty_pair;
        }
 
    }
 
    // cpu-based uncompression algorithm for VLQ base64 compressed dict
    // only workable before init VLQ base64 runtime dict
    std::pair<std::vector<uint64_t>, std::vector<IdType> > mget_VLQ_base64_binary_codes(boost::python::list& keys) {
        std::vector<uint64_t> binary_codes(0);
        std::vector<IdType> id_vector(0);
        std::pair<std::vector<uint64_t>, std::vector<IdType> > return_pair(binary_codes, id_vector);

        for (int i = 0; i < len(keys); i++) {
            std::pair<std::vector<uint64_t>, std::vector<IdType> > partial_binary_codes = get_VLQ_base64_binary_codes(boost::python::extract<uint32_t>(keys[i]));

            BOOST_FOREACH(uint64_t binary_code, partial_binary_codes.first) {
                return_pair.first.insert(return_pair.first.end(), binary_code);
            }
            BOOST_FOREACH(IdType binary_code_id, partial_binary_codes.second) {
                return_pair.second.insert(return_pair.second.end(), binary_code_id);
            }
        }
        return return_pair;
    }
  
    static const uint8_t VLQ_BASE_SHIFT = 5;
    static const uint8_t VLQ_BASE = 1 << VLQ_BASE_SHIFT;
    static const uint8_t VLQ_BASE_MASK = VLQ_BASE - 1;
    static const uint8_t VLQ_CONTINUATION_BIT = VLQ_BASE;

    // encoding/decoding VLQ base64
    std::string base64VLQ_encode(BitCountType val) {
        std::string encoded = "";
        BitCountType digit;

        do {
            digit = val & VLQ_BASE_MASK;
            val >>= VLQ_BASE_SHIFT;
            if (val > 0) {
                digit |= VLQ_CONTINUATION_BIT;
            }
            encoded += base64_chars[digit];
        } while (val > 0);

        return encoded;
    }


    std::vector<BitCountType> base64VLQ_decode(std::string str) {
        uint32_t i = 0;
        uint32_t strLen = str.length();
        std::vector<BitCountType> results(0);

        while (i < strLen) {
            BitCountType result = 0;
            uint8_t shift = 0;
            uint8_t continuation, digit;
            do {
                if (i >= strLen) {
                    throw new std::string("Expected more digits in base 64 VLQ value.");
                }
                digit = base64_chars.find(str[i++]);
                continuation = digit & VLQ_CONTINUATION_BIT;    
                digit &= VLQ_BASE_MASK;
                result = result + (digit << shift);
                shift += VLQ_BASE_SHIFT;
            } while (continuation > 0);
            results.insert(results.end(), result);
        }

        return results;
    }
 
    std::pair<BitCountType, uint32_t> incre_base64VLQ_decode(std::string str, uint32_t offset) {
        uint32_t i = offset;
        uint32_t strLen = str.length();

        BitCountType result = 0;
        uint8_t shift = 0;
        uint8_t continuation, digit;

        do {
            if (i >= strLen) {
                throw new std::string("Expected more digits in base 64 VLQ value.");
            }
            digit = base64_chars.find(str[i++]);
            continuation = digit & VLQ_CONTINUATION_BIT;    
            digit &= VLQ_BASE_MASK;
            result = result + (digit << shift);
            shift += VLQ_BASE_SHIFT;
        } while (continuation > 0);

        std::pair<BitCountType, uint32_t> decode_pair(result, i);
        return decode_pair;
    }
 
    int get_dict_status() { return dict_status; }    

    // status code for dict
    // -1: not initialized
    // 0: compressed dict
    // 1: VLQ base64 dict           # from 0 by to_VLQ_base64_dict
    // 2: runtime dict              # from 0 by init_runtime_dict
    // 3: VLQ base64 runtime dict   # from 1 by init_runtime_VLQ_base64_dict
    int dict_status;

    std::map<std::vector<uint8_t>, std::pair<std::vector<std::vector<BitCountType> >, std::vector<IdType> > > column_dict;

    std::map<std::vector<uint8_t>, std::pair<std::vector<std::string>, std::vector<IdType> > > column_vlq_dict;
 
    std::map<std::vector<uint8_t>, std::pair<std::vector<uint32_t>, std::pair<std::vector<BitCountType*>, std::vector<IdType> > > > runtime_dict;

    std::map<std::vector<uint8_t>, std::pair<std::vector<uint32_t>, std::pair<std::vector<char*>, std::vector<IdType> > > > runtime_vlq_dict;
 
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
    oa << dict.column_vlq_dict;
    oa << dict.dict_status;
}

template <class BitCountType, class IdType>
void load_compress(char* filename, FastCompressDict<BitCountType, IdType>& dict) {
    std::ifstream ifs(filename);

    boost::archive::text_iarchive ia(ifs);
    ia >> dict.dict;
    ia >> dict.key_dimensions;
    ia >> dict.index_key_dimension;
    ia >> dict.column_dict;
    ia >> dict.column_vlq_dict;
    ia >> dict.dict_status;
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
 
    to_python_converter<
        std::vector<std::vector<PyObject*> >,
        vector_pyobject_vector_to_python>();
 

    class_<FastDict<std::string> >("FastDict", init<uint8_t>())
        .def("get", &FastDict<std::string>::get)
        .def("mget", &FastDict<std::string>::mget)
        .def("set", &FastDict<std::string>::set)
        .def("append", &FastDict<std::string>::append)
        .def("batch_append", &FastDict<std::string>::batch_append)
        .def("batch_iter_append", &FastDict<std::string>::batch_iter_append)
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
        .def("mget", &FastDict<uint32_t>::mget)
        .def("set", &FastDict<uint32_t>::set)
        .def("append", &FastDict<uint32_t>::append)
        .def("batch_append", &FastDict<uint32_t>::batch_append)
        .def("batch_iter_append", &FastDict<uint32_t>::batch_iter_append)
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
        .def("mget", &FastCompressDict<uint8_t, uint32_t>::mget)
        .def("set", &FastCompressDict<uint8_t, uint32_t>::set)
        .def("append", &FastCompressDict<uint8_t, uint32_t>::append)
        .def("batch_append", &FastCompressDict<uint8_t, uint32_t>::batch_append)
        .def("batch_iter_append", &FastCompressDict<uint8_t, uint32_t>::batch_iter_append)
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
        .def("mget_binary_codes", &FastCompressDict<uint8_t, uint32_t>::mget_binary_codes)
        .def("get_cols_as_buffer", &FastCompressDict<uint8_t, uint32_t>::get_cols_as_buffer)
        .def("mget_cols_as_buffer", &FastCompressDict<uint8_t, uint32_t>::mget_cols_as_buffer)
        .def("get_image_ids", &FastCompressDict<uint8_t, uint32_t>::get_image_ids)
        .def("mget_image_ids", &FastCompressDict<uint8_t, uint32_t>::mget_image_ids)
        .def("get_image_ids_before_runtime_init", &FastCompressDict<uint8_t, uint32_t>::get_image_ids_before_runtime_init)
        .def("mget_image_ids_before_runtime_init", &FastCompressDict<uint8_t, uint32_t>::mget_image_ids_before_runtime_init)
        .def("get_VLQ_base64_image_ids_before_runtime_init", &FastCompressDict<uint8_t, uint32_t>::get_VLQ_base64_image_ids_before_runtime_init)
        .def("mget_VLQ_base64_image_ids_before_runtime_init", &FastCompressDict<uint8_t, uint32_t>::mget_VLQ_base64_image_ids_before_runtime_init)
        .def("init_runtime_dict", &FastCompressDict<uint8_t, uint32_t>::init_runtime_dict)
        .def("base64VLQ_encode", &FastCompressDict<uint8_t, uint32_t>::base64VLQ_encode)
        .def("base64VLQ_decode", &FastCompressDict<uint8_t, uint32_t>::base64VLQ_decode)
        .def("to_VLQ_base64_dict", &FastCompressDict<uint8_t, uint32_t>::to_VLQ_base64_dict)
        .def("init_runtime_VLQ_base64_dict", &FastCompressDict<uint8_t, uint32_t>::init_runtime_VLQ_base64_dict) 
        .def("get_VLQ_base64_cols_as_buffer", &FastCompressDict<uint8_t, uint32_t>::get_VLQ_base64_cols_as_buffer
)
        .def("mget_VLQ_base64_cols_as_buffer", &FastCompressDict<uint8_t, uint32_t>::mget_VLQ_base64_cols_as_buffer
)
        .def("get_VLQ_base64_image_ids", &FastCompressDict<uint8_t, uint32_t>::get_VLQ_base64_image_ids)
        .def("mget_VLQ_base64_image_ids", &FastCompressDict<uint8_t, uint32_t>::mget_VLQ_base64_image_ids)
        .def("get_VLQ_base64_cols", &FastCompressDict<uint8_t, uint32_t>::get_VLQ_base64_cols)
        .def("get_dict_status", &FastCompressDict<uint8_t, uint32_t>::get_dict_status)
        .def("get_VLQ_base64_binary_codes", &FastCompressDict<uint8_t, uint32_t>::get_VLQ_base64_binary_codes)
        .def("mget_VLQ_base64_binary_codes", &FastCompressDict<uint8_t, uint32_t>::mget_VLQ_base64_binary_codes)
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
        .def("mget", &FastCompressDict<uint32_t, uint32_t>::mget)
        .def("set", &FastCompressDict<uint32_t, uint32_t>::set)
        .def("append", &FastCompressDict<uint32_t, uint32_t>::append)
        .def("batch_append", &FastCompressDict<uint32_t, uint32_t>::batch_append)
        .def("batch_iter_append", &FastCompressDict<uint32_t, uint32_t>::batch_iter_append)
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
        .def("mget_binary_codes", &FastCompressDict<uint32_t, uint32_t>::mget_binary_codes)
        .def("get_cols_as_buffer", &FastCompressDict<uint32_t, uint32_t>::get_cols_as_buffer)
        .def("mget_cols_as_buffer", &FastCompressDict<uint32_t, uint32_t>::mget_cols_as_buffer)
        .def("get_image_ids", &FastCompressDict<uint32_t, uint32_t>::get_image_ids)
        .def("mget_image_ids", &FastCompressDict<uint32_t, uint32_t>::mget_image_ids)
        .def("get_image_ids_before_runtime_init", &FastCompressDict<uint32_t, uint32_t>::get_image_ids_before_runtime_init)
        .def("mget_image_ids_before_runtime_init", &FastCompressDict<uint32_t, uint32_t>::mget_image_ids_before_runtime_init)
        .def("get_VLQ_base64_image_ids_before_runtime_init", &FastCompressDict<uint32_t, uint32_t>::get_VLQ_base64_image_ids_before_runtime_init)
        .def("mget_VLQ_base64_image_ids_before_runtime_init", &FastCompressDict<uint32_t, uint32_t>::mget_VLQ_base64_image_ids_before_runtime_init)
        .def("init_runtime_dict", &FastCompressDict<uint32_t, uint32_t>::init_runtime_dict)
        .def("base64VLQ_encode", &FastCompressDict<uint32_t, uint32_t>::base64VLQ_encode)
        .def("base64VLQ_decode", &FastCompressDict<uint32_t, uint32_t>::base64VLQ_decode)
        .def("to_VLQ_base64_dict", &FastCompressDict<uint32_t, uint32_t>::to_VLQ_base64_dict)
        .def("init_runtime_VLQ_base64_dict", &FastCompressDict<uint32_t, uint32_t>::init_runtime_VLQ_base64_dict)
        .def("get_VLQ_base64_cols_as_buffer", &FastCompressDict<uint32_t, uint32_t>::get_VLQ_base64_cols_as_buffer)
        .def("mget_VLQ_base64_cols_as_buffer", &FastCompressDict<uint32_t, uint32_t>::mget_VLQ_base64_cols_as_buffer)
        .def("get_VLQ_base64_image_ids", &FastCompressDict<uint32_t, uint32_t>::get_VLQ_base64_image_ids)
        .def("mget_VLQ_base64_image_ids", &FastCompressDict<uint32_t, uint32_t>::mget_VLQ_base64_image_ids)
        .def("get_VLQ_base64_cols", &FastCompressDict<uint32_t, uint32_t>::get_VLQ_base64_cols)
        .def("get_dict_status", &FastCompressDict<uint32_t, uint32_t>::get_dict_status)
        .def("get_VLQ_base64_binary_codes", &FastCompressDict<uint32_t, uint32_t>::get_VLQ_base64_binary_codes)
        .def("mget_VLQ_base64_binary_codes", &FastCompressDict<uint32_t, uint32_t>::mget_VLQ_base64_binary_codes)
    ;

    class_<std::vector<std::vector<uint32_t> > >("CompressedUInt32ColumnIntVec")
        .def(vector_indexing_suite<std::vector<std::vector<uint32_t> > >())
    ;

    class_<std::vector<uint32_t> >("UInt32BitCountVec")
        .def(vector_indexing_suite<std::vector<uint32_t> >())
    ;
 
    class_<std::vector<std::string> >("StringVec")
        .def(vector_indexing_suite<std::vector<std::string> >())
    ;
 
    class_<std::pair<std::vector<std::vector<uint32_t> >, std::vector<uint32_t> > >("CompressedUInt32ColumnsIdsIntPair")
        .def_readwrite("first", &std::pair<std::vector<std::vector<uint32_t> >, std::vector<uint32_t> >::first)
        .def_readwrite("second", &std::pair<std::vector<std::vector<uint32_t> >, std::vector<uint32_t> >::second)
    ;

    class_<std::pair<std::vector<std::string>, std::vector<uint32_t> > >("CompressedStringColumnsIdsIntPair")
        .def_readwrite("first", &std::pair<std::vector<std::string>, std::vector<uint32_t> >::first)
        .def_readwrite("second", &std::pair<std::vector<std::string>, std::vector<uint32_t> >::second)
    ; 
 
    //class_<std::vector<std::vector<PyObject*> > >("ColBuffersVec")
    //    .def(vector_indexing_suite<std::vector<std::vector<PyObject*> > >())
    //;
 
    def("save_compress_uint32_int", save_compress<uint32_t, uint32_t>);
    def("load_compress_uint32_int", load_compress<uint32_t, uint32_t>);
 
}


#include <string>
#include <map>
#include <vector>
#include <list>
#include <utility>
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


template <class IdType>
class FastDict
{

public:
    
    FastDict() {}

    friend class boost::serialization::access;

    void set(uint32_t key, uint64_t hash_key, IdType id) {
        std::pair<uint64_t, IdType> element(hash_key, id);
        std::vector<std::pair<uint64_t, IdType> > element_list(1, element);
        dict[key] = element_list;
    }

    std::vector<std::pair<uint64_t, IdType> > get(uint32_t key) {
        if (dict.count(key) > 0)
            return dict[key];
        else {
            std::pair<uint64_t, IdType> element(0, *new IdType());
            std::vector<std::pair<uint64_t, IdType> > element_list(1, element);
            return element_list;
        }
    }

    bool exist(uint32_t key) {
        if (dict.count(key) > 0)
            return true;
        else
            return false;
    }

    void clear() { dict.clear(); }

    void merge(FastDict<IdType>& source) {

        std::pair<uint32_t, std::vector<std::pair<uint64_t, IdType> > > me;
        std::vector<uint32_t> keys;
        BOOST_FOREACH(me, source.dict) {
            std::pair<uint64_t, IdType> element;
            BOOST_FOREACH(element, me.second) {
                dict[me.first].insert(dict[me.first].begin(), element);
            }
        }

    }

    uint32_t size() { return dict.size(); }

    void append(uint32_t key, uint64_t hash_key, IdType id) {
        std::pair<uint64_t, IdType> element(hash_key, id);
        dict[key].insert(dict[key].begin(), element);
    }

    std::vector<uint32_t> keys() {
        std::pair<uint32_t, std::vector<std::pair<uint64_t, IdType> > > me;
        std::vector<uint32_t> keys;
        BOOST_FOREACH(me, dict) {
            keys.push_back(me.first);
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

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & dict;
        ar & key_dimensions;
    }

    std::map<uint32_t, std::vector<std::pair<uint64_t, IdType> > > dict;
    std::vector<uint32_t> key_dimensions;
};

template <class IdType>
void save(char* filename, FastDict<IdType> dict) {
   std::ofstream ofs(filename);

   boost::archive::text_oarchive oa(ofs);
   oa << dict.dict;
   oa << dict.key_dimensions;
}

template <class IdType>
void load(char* filename, FastDict<IdType>& dict) {
    std::ifstream ifs(filename);

    boost::archive::text_iarchive ia(ifs);
    ia >> dict.dict;
    ia >> dict.key_dimensions;

}

using namespace boost::python;

BOOST_PYTHON_MODULE(fastdict)
{
    class_<FastDict<std::string> >("FastDict")
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

    class_<FastDict<uint64_t> >("FastIntDict")
        .def("get", &FastDict<uint64_t>::get)
        .def("set", &FastDict<uint64_t>::set)
        .def("append", &FastDict<uint64_t>::append)
        .def("size", &FastDict<uint64_t>::size)
        .def("keys", &FastDict<uint64_t>::keys)
        .def("set_keydimensions", &FastDict<uint64_t>::set_keydimensions)
        .def("get_keydimensions", &FastDict<uint64_t>::get_keydimensions)
        .def("exist", &FastDict<uint64_t>::exist)
        .def("clear", &FastDict<uint64_t>::clear)
        .def("merge", &FastDict<uint64_t>::merge)
    ;

    class_<std::vector<std::pair<uint64_t, uint64_t> > >("PairIntVec")
        .def(vector_indexing_suite<std::vector<std::pair<uint64_t, uint64_t> > >())
    ;                                                           

    class_<std::pair<uint64_t, uint64_t> >("HashIntPair")
        .def_readwrite("first", &std::pair<uint64_t, uint64_t>::first)
        .def_readwrite("second", &std::pair<uint64_t, uint64_t>::second)
    ;

    def("save_int", save<uint64_t>);
    def("load_int", load<uint64_t>);
}


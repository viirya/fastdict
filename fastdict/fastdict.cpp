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


class FastDict
{

public:
    
    FastDict() {}

    friend class boost::serialization::access;

    void set(uint32_t key, uint64_t hash_key, std::string id) {
        std::pair<uint64_t, std::string> element(hash_key, id);
        std::vector<std::pair<uint64_t, std::string> > element_list(1, element);
        dict[key] = element_list;
    }

    std::vector<std::pair<uint64_t, std::string> > get(uint32_t key) {
        if (dict.count(key) > 0)
            return dict[key];
        else {
            std::pair<uint64_t, std::string> element(0, "");
            std::vector<std::pair<uint64_t, std::string> > element_list(1, element);
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

    void merge(FastDict& source) {

        std::pair<uint32_t, std::vector<std::pair<uint64_t, std::string> > > me;
        std::vector<uint32_t> keys;
        BOOST_FOREACH(me, source.dict) {
            std::pair<uint64_t, std::string> element;
            BOOST_FOREACH(element, me.second) {
                dict[me.first].insert(dict[me.first].begin(), element);
            }
        }

    }

    uint32_t size() { return dict.size(); }

    void append(uint32_t key, uint64_t hash_key, std::string id) {
        std::pair<uint64_t, std::string> element(hash_key, id);
        dict[key].insert(dict[key].begin(), element);
    }

    std::vector<uint32_t> keys() {
        std::pair<uint32_t, std::vector<std::pair<uint64_t, std::string> > > me;
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

    std::map<uint32_t, std::vector<std::pair<uint64_t, std::string> > > dict;
    std::vector<uint32_t> key_dimensions;
};

void save(char* filename, FastDict dict) {
   std::ofstream ofs(filename);

   boost::archive::text_oarchive oa(ofs);
   oa << dict.dict;
   oa << dict.key_dimensions;
}

void load(char* filename, FastDict& dict) {
    std::ifstream ifs(filename);

    boost::archive::text_iarchive ia(ifs);
    ia >> dict.dict;
    ia >> dict.key_dimensions;

}

using namespace boost::python;

BOOST_PYTHON_MODULE(fastdict)
{
    class_<FastDict>("FastDict")
        .def("get", &FastDict::get)
        .def("set", &FastDict::set)
        .def("append", &FastDict::append)
        .def("size", &FastDict::size)
        .def("keys", &FastDict::keys)
        .def("set_keydimensions", &FastDict::set_keydimensions)
        .def("get_keydimensions", &FastDict::get_keydimensions)
        .def("exist", &FastDict::exist)
        .def("clear", &FastDict::clear)
        .def("merge", &FastDict::merge)
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
 

    def("save", save);
    def("load", load);
}



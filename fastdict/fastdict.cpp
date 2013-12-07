#include <string>
#include <map>
#include <vector>
#include <utility>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>

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

    std::vector<std::pair<uint64_t, std::string> > get(uint32_t key) { return dict[key]; }
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

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & dict;
    }

    std::map<uint32_t, std::vector<std::pair<uint64_t, std::string> > > dict;
};

void save(char* filename, FastDict dict) {
   std::ofstream ofs(filename);

   boost::archive::text_oarchive oa(ofs);
   oa << dict.dict;
}

void load(char* filename, FastDict& dict) {
    std::ifstream ifs(filename);

    boost::archive::text_iarchive ia(ifs);
    ia >> dict.dict;

}

#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(fastdict)
{
    class_<FastDict>("FastDict")
        .def("get", &FastDict::get)
        .def("set", &FastDict::set)
        .def("append", &FastDict::append)
        .def("size", &FastDict::size)
        .def("keys", &FastDict::keys)
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



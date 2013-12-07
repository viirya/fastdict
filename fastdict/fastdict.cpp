#include <string>
#include <map>
#include <vector>
#include <utility>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>

struct FastDict
{
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

    std::map<uint32_t, std::vector<std::pair<uint64_t, std::string> > > dict;
};

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

}



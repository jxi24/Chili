#pragma once

#include "spdlog/spdlog.h"
#include <functional>
#include <map>
#include <memory>
#include <iostream>

namespace apes {

template<typename Base>
class Factory {
    using deserialize_func = std::function<std::unique_ptr<Base>(std::istream&)>;

    static std::map<std::string, deserialize_func>& Registry() {
        static std::map<std::string, deserialize_func> registry;
        return registry;
    }

    public:
        static std::unique_ptr<Base> Deserialize(std::istream &in) {
            size_t size;
            in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
            char *data = new char[size+1];
            in.read(data, size);
            data[size] = '\0';
            std::string name = data;
            delete[] data;

            auto deserialize = Registry().at(name);
            return deserialize(in);
        }

        template<class Derived>
        static void Register(const std::string &name) {
            if(IsRegistered(name))
                spdlog::error("{} is already registered!", name);
            Registry()[name] = Derived::_Deserialize;
        }

        static bool IsRegistered(const std::string &name) {
            return Registry().find(name) != Registry().end();
        }

        static void Display() {
            fmt::print("Registered {}:\n", Base::Name());
            for(const auto &registered : Registry())
                fmt::print("  - {}\n", registered.first);
        }
};

template<typename Base, typename Derived>
class Registrable {
    protected:
        Registrable() = default;
        virtual ~Registrable() {
            if(!m_registered)
                spdlog::error("Error registering");
        }

        static bool Register() {
            Factory<Base>::template Register<Derived>(Derived::Name());
            return true;
        }

    private:
        static const bool m_registered;
};
template<typename Base, typename Derived>
const bool Registrable<Base, Derived>::m_registered = Registrable<Base, Derived>::Register();

}

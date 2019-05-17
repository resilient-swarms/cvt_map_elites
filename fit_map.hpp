#ifndef CVT_MAP_ELITES_FIT_MAP_HPP_
#define CVT_MAP_ELITES_FIT_MAP_HPP_

#include <sferes/fit/fitness.hpp>

#define FIT_MAP(Name) SFERES_FITNESS(Name, sferes::fit::FitMap)

namespace sferes {
    namespace fit {
        SFERES_FITNESS(FitMap, sferes::fit::Fitness)
        {
        public:
            FitMap() : _desc(Params::ea::number_of_dimensions, -9.9f) {}

            const std::vector<float>& desc() const { return _desc; }

            void set_desc(std::vector<float> & x)
            {
                assert(x.size() == Params::ea::number_of_dimensions);
                for (size_t i = 0; i < x.size(); ++i)
                    assert(x[i] >= 0.0 && x[i] <= 1.0);
                _desc = x;
            }

        protected:
            std::vector<float> _desc;
        };
    }
}

#endif

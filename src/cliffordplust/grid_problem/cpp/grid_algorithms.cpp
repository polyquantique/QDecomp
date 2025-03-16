/* Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
 * 
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 * 
 *        http://www.apache.org/licenses/LICENSE-2.0
 * 
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

#ifndef GRID_ALGORITHMS_HPP
#define GRID_ALGORITHMS_HPP

#include <cmath>
#include <vector>

#include "..\..\rings\cpp\Rings.hpp"


inline const Zsqrt2 LAMBDA(1, 1);
inline const Zsqrt2 LAMBDA_INV(-1, 1);


/**
 * @class GridProblem1D
 * @brief A class to solve a 1D grid problem
 * 
 * This class solves a 1D grid problem. It returns an iterator that generates all the solutions
 * of the problem.
 */
class GridProblem1D {
    private:
        double _amin, _amax;  ///< Bounds of the first interval
        double _bmin, _bmax;  ///< Bounds of the second interval

        double _amin_, _amax_;  ///< Bounds of the scaled first interval
        double _bmin_, _bmax_;  ///< Bounds of the scaled second interval

        long long int _qmin, _qmax;  ///< Bounds of the q coefficient
        int _n_scaling;              ///< Exponent used to scale the grid problem

    public:
        /**
         * @brief Construct a new Grid Problem 1D object
         * 
         * @param amin The minimum bound of the first interval
         * @param amax The maximum bound of the first interval
         * @param bmin The minimum bound of the second interval
         * @param bmax The maximum bound of the second interval
         * @throw std::invalid_argument if amin > amax or bmin > bmax
         */
        GridProblem1D(double amin, double amax, double bmin, double bmax) :
        _amin(amin), _amax(amax), _bmin(bmin), _bmax(bmax)
        {
            // Check if the bounds are in increasing order
            if ((amin > amax) or (bmin > bmax)) {
                throw std::invalid_argument("The interval bounds must be in increasing order. Got ["
                    + std::to_string(amin) + ", " + std::to_string(amax) + "] and ["
                    + std::to_string(bmin) + ", " + std::to_string(bmax) + "].");
            }

            // Calculate the scaled intervals
            double delta = amax - amin;
            _n_scaling = (unsigned short int)std::floor(
                log(LAMBDA.to_float() * delta) / log(LAMBDA.to_float())
            );
            _amin_ = amin * std::pow(LAMBDA.to_float(), -_n_scaling);
            _amax_ = amax * std::pow(LAMBDA.to_float(), -_n_scaling);
            _bmin_ = bmin * std::pow(-LAMBDA.to_float(), _n_scaling);
            _bmax_ = bmax * std::pow(-LAMBDA.to_float(), _n_scaling);

            if (_n_scaling & 1) {  // The interval is inverted in this case
                double temp = _bmin_;
                _bmin_ = _bmax_;
                _bmax_ = temp;
            }

            // Calculate the interval of values that the q coefficient can take
            double qmin_f = (_amin_ - _bmax_) / std::sqrt(8);
            double qmax_f = (_amax_ - _bmin_) / std::sqrt(8);

            // Convert the floating bounds to integer bounds
            // To avoid floating-point errors that may result in forgetting a solution, we add a
            // small epsilon to the bounds
            _qmin = static_cast<long long int>(std::floor(qmin_f - 1e-6)) + 1;
            _qmax = static_cast<long long int>(std::floor(qmax_f + 1e-6));
        }

        /**
         * @class Iterator
         * @brief An iterator to generate the solutions of the grid problem
         */
        class Iterator {
            private:
                double _amin, _amax;    ///< Bounds of the first interval
                double _amin_, _amax_;  ///< Bounds of the scaled first interval

                double _bmin, _bmax;  ///< Bounds of the second interval

                long long int _p_last, _q_last;  ///< Coefficients of the last solution
                long long int _p, _q;            ///< Current solution
                long long int _qmin, _qmax;      ///< Bounds for the q coefficient
                int _n_scaling;                  ///< Exponent used to scale the grid problem

            public:
                /**
                 * @brief Construct a new Iterator object
                 * 
                 * @param amin The minimum bound of the first interval
                 * @param amax The maximum bound of the first interval
                 * @param bmin The minimum bound of the second interval
                 * @param bmax The maximum bound of the second interval
                 * @param amin_ The minimum bound of the scaled first interval
                 * @param amax_ The maximum bound of the scaled first interval
                 * @param qmin The minimum bound of the q coefficient
                 * @param qmax The maximum bound of the q coefficient
                 * @param n_scaling The exponent used to scale the grid problem
                 */
                Iterator(
                    double amin, double amax, double bmin, double bmax,
                    double amin_, double amax_, long long int qmin, long long int qmax,
                    int n_scaling
                ) : _amin(amin), _amax(amax), _amin_(amin_), _amax_(amax_),
                    _bmin(bmin), _bmax(bmax), _q(qmin), _qmin(qmin), _qmax(qmax),
                    _n_scaling(n_scaling)
                {
                    // Find the first solution
                    _q--;
                    operator++();
                }

                /**
                 * @brief Get the current solution
                 */
                Zsqrt2 operator*() const { return Zsqrt2(_p_last, _q_last); }

                /**
                 * @brief Increment the iterator
                 */
                Iterator& operator++() {
                    while (_q <= _qmax) {
                        // Increment the solution
                        _q++;
                    
                        // Calculate the interval of values that the p coefficient can take
                        // To avoid floating-point errors that may result in forgetting a solution,
                        // we add a small epsilon to the bounds
                        double pmin_f = _amin_ - (double)_q * std::sqrt(2) - 1.0e-6;
                        double pmax_f = _amax_ - (double)_q * std::sqrt(2) + 1.0e-6;

                        // Determine if there is an integer solution in the p interval
                        if (std::floor(pmin_f) != std::floor(pmax_f)) {
                            _p = static_cast<long long int>(std::floor(pmax_f));

                            Zsqrt2 alpha_(_p, _q);  // Scaled solution

                            // Unscaled solution
                            Zsqrt2 alpha(0, 0);
                            if (_n_scaling < 0) {
                                alpha = alpha_ * LAMBDA_INV.pow((unsigned short int)-_n_scaling);
                            } else {
                                alpha = alpha_ * LAMBDA.pow((unsigned short int)_n_scaling);
                            }

                            // Check if the unscaled solution is in the original interval
                            double alpha_f = alpha.to_float();
                            double alpha_conj_f = alpha.sqrt2_conjugate().to_float();

                            if ((alpha_f >= _amin) and (alpha_f <= _amax) and
                                (alpha_conj_f >= _bmin) and (alpha_conj_f <= _bmax)) {
                                    _p_last = alpha.p();
                                    _q_last = alpha.q();
                                    return *this;
                            }
                        }
                    }
                    return *this;
                }

                /**
                 * @brief Used to determine wether the iterator is at the end
                 * 
                 * @param _ The other iterator (unused)
                 * @return true If this iterator is at the end
                 * @return false If this iterators are not at the end
                 */
                bool operator!=(const Iterator& _) const { return _q <= _qmax; }
        };

        /**
         * @brief Get the beginning of the iterator
         * 
         * @return Iterator The beginning of the iterator
         */
        Iterator begin() {
            return Iterator(_amin, _amax, _bmin, _bmax, _amin_, _amax_, _qmin, _qmax, _n_scaling);
        }

        /**
         * @brief Get the end of the iterator
         * 
         * This function is called when initializing the iterator in a for loop. It is however never
         * properly used since the for loop only uses it with the != operator which discards its
         * argument. Since the result of this method is useless, it returns the same iterator as the
         * one returned by the begin() method.
         * 
         * @return Iterator The end of the iterator
         */
        Iterator end() {
            return Iterator(_amin, _amax, _bmin, _bmax, _amin_, _amax_, _qmin, _qmax, _n_scaling);
        }
};




#endif // GRID_ALGORITHMS_HPP

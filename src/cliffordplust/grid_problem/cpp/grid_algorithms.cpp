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

#include "..\..\rings\cpp\Rings.hpp"


const long double SQRT2 = std::sqrt(static_cast<long double>(2));
const long double SQRT2_INV = 1.0 / SQRT2;
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
        long double _amin, _amax;  ///< Bounds of the first interval
        long double _bmin, _bmax;  ///< Bounds of the second interval

        long double _amin_, _amax_;  ///< Bounds of the scaled first interval
        long double _bmin_, _bmax_;  ///< Bounds of the scaled second interval

        long long int _qmin, _qmax;  ///< Bounds of the q coefficient
        int _n_scaling;              ///< Exponent used to scale the grid problem

    public:
        /**
         * @brief Construct a new GridProblem1D object
         * 
         * @param amin The minimum bound of the first interval
         * @param amax The maximum bound of the first interval
         * @param bmin The minimum bound of the second interval
         * @param bmax The maximum bound of the second interval
         * @throw std::invalid_argument if amin > amax or bmin > bmax
         */
        GridProblem1D(long double amin, long double amax, long double bmin, long double bmax) :
        _amin(amin), _amax(amax), _bmin(bmin), _bmax(bmax)
        {
            // Check if the bounds are in increasing order
            if ((amin > amax) or (bmin > bmax)) {
                throw std::invalid_argument("The interval bounds must be in increasing order. Got ["
                    + std::to_string(amin) + ", " + std::to_string(amax) + "] and ["
                    + std::to_string(bmin) + ", " + std::to_string(bmax) + "].");
            }

            // Calculate the scaled intervals
            long double delta = amax - amin;
            _n_scaling = (int)std::floor(
                std::log(LAMBDA.to_long_double() * delta) / std::log(LAMBDA.to_long_double())
            );
            _amin_ = amin * std::pow(LAMBDA.to_long_double(), static_cast<long double>(-_n_scaling));
            _amax_ = amax * std::pow(LAMBDA.to_long_double(), static_cast<long double>(-_n_scaling));
            _bmin_ = bmin * std::pow(-LAMBDA.to_long_double(), static_cast<long double>(_n_scaling));
            _bmax_ = bmax * std::pow(-LAMBDA.to_long_double(), static_cast<long double>(_n_scaling));

            if (_n_scaling & 1) {  // The interval is inverted in this case
                long double temp = _bmin_;
                _bmin_ = _bmax_;
                _bmax_ = temp;
            }

            // Calculate the interval of values that the q coefficient can take
            long double qmin_f = (_amin_ - _bmax_) / std::sqrt(static_cast<long double>(8.0));
            long double qmax_f = (_amax_ - _bmin_) / std::sqrt(static_cast<long double>(8.0));

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
                long double _amin, _amax;    ///< Bounds of the first interval
                long double _amin_, _amax_;  ///< Bounds of the scaled first interval

                long double _bmin, _bmax;  ///< Bounds of the second interval

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
                    long double amin, long double amax, long double bmin, long double bmax,
                    long double amin_, long double amax_, long long int qmin, long long int qmax,
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
                        long double pmin_f = _amin_ - (long double)_q * std::sqrt(static_cast<long double>(2)) - 1.0e-12;
                        long double pmax_f = _amax_ - (long double)_q * std::sqrt(static_cast<long double>(2)) + 1.0e-12;

                        // Determine if there is an integer solution in the p interval
                        if (std::floor(pmin_f) != std::floor(pmax_f)) {
                            _p = static_cast<long long int>(std::floor(pmax_f));

                            Zsqrt2 alpha_(_p, _q);  // Scaled solution

                            // Unscaled solution
                            Zsqrt2 alpha(0, 0);
                            if (_n_scaling < 0) {
                                alpha = alpha_ * LAMBDA_INV.pow(static_cast<unsigned short int>(-_n_scaling));
                            } else {
                                alpha = alpha_ * LAMBDA.pow(static_cast<unsigned short int>(_n_scaling));
                            }

                            // Check if the unscaled solution is in the original interval
                            long double alpha_f = alpha.to_long_double();
                            long double alpha_conj_f = alpha.sqrt2_conjugate().to_long_double();

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
                bool operator!=(const Iterator&) const { return _q <= _qmax; }
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

/**
 * @class GridProblem2D
 * @brief A class to solve a 2D grid problem
 * 
 * This class solves a 2D grid problem. It returns an iterator that generates all the solutions
 * of the problem. Solving this problem is equivalent to solve two times two 1D grid problems and
 * returning the Cartesian product of the solutions (4 grid problems are solved in total).
 */
class GridProblem2D{
    private:
        GridProblem1D _gp_x1;  ///< First grid problem for the first coordinate
        GridProblem1D _gp_y1;  ///< First grid problem for the second coordinate
        GridProblem1D _gp_x2;  ///< Second grid problem for the first coordinate
        GridProblem1D _gp_y2;  ///< Second grid problem for the second coordinate

    public:
        /**
         * @brief Construct a new GridProblem2D object
         * 
         * @param axmin The minimum bound of the first interval for the first coordinate
         * @param axmax The maximum bound of the first interval for the first coordinate
         * @param bxmin The minimum bound of the second interval for the first coordinate
         * @param bxmax The maximum bound of the second interval for the first coordinate
         * @param aymin The minimum bound of the first interval for the second coordinate
         * @param aymax The maximum bound of the first interval for the second coordinate
         * @param bymin The minimum bound of the second interval for the second coordinate
         * @param bymax The maximum bound of the second interval for the second coordinate
         * @throw std::invalid_argument if axmin > axmax, bxmin > bxmax, aymin > aymax or bymin > bymax
         */
        GridProblem2D(
            long double axmin, long double axmax, long double bxmin, long double bxmax,
            long double aymin, long double aymax, long double bymin, long double bymax
        ) : _gp_x1(axmin, axmax, bxmin, bxmax), _gp_y1(aymin, aymax, bymin, bymax),
            _gp_x2(axmin - SQRT2_INV, axmax - SQRT2_INV, bxmin + SQRT2_INV, bxmax + SQRT2_INV),
            _gp_y2(aymin - SQRT2_INV, aymax - SQRT2_INV, bymin + SQRT2_INV, bymax + SQRT2_INV)
        {
            // Check if the bounds are in increasing order
            if ((axmin > axmax) or (aymin > aymax) or (bxmin > bxmax) or (bymin > bymax)) {
                throw std::invalid_argument("The interval bounds must be in increasing order. Got ["
                    + std::to_string(axmin) + ", " + std::to_string(axmax) + "], ["
                    + std::to_string(aymin) + ", " + std::to_string(aymax) + "], ["
                    + std::to_string(bxmin) + ", " + std::to_string(bxmax) + "] and ["
                    + std::to_string(bymin) + ", " + std::to_string(bymax) + "].");
            }
        }
        
        /**
         * @class Iterator
         * @brief An iterator to generate the solutions of the grid problem
         */
        class Iterator {
            private:
                bool _first_completed = false;   ///< Flag to indicate if the first problem is completed
                bool _second_completed = false;  ///< Flag to indicate if the second problem is completed

                GridProblem1D::Iterator _x_it1;  ///< Iterator for the first coordinate for the first problem
                GridProblem1D::Iterator _y_it1;  ///< Iterator for the second coordinate for the first problem
                GridProblem1D::Iterator _x_it2;  ///< Iterator for the first coordinate for the second problem
                GridProblem1D::Iterator _y_it2;  ///< Iterator for the second coordinate for the second problem

                const GridProblem1D::Iterator _y_it1_begin;  ///< Beginning of the y iterator for the first problem
                const GridProblem1D::Iterator _y_it2_begin;  ///< Beginning of the y iterator for the second problem
            
                const GridProblem1D::Iterator _x_it1_end;  ///< End of the x iterator for the first problem
                const GridProblem1D::Iterator _y_it1_end;  ///< End of the y iterator for the first problem
                const GridProblem1D::Iterator _x_it2_end;  ///< End of the x iterator for the second problem
                const GridProblem1D::Iterator _y_it2_end;  ///< End of the y iterator for the second problem

            public:
                /**
                 * @brief Construct a new Iterator object
                 * 
                 * @param x_it1_begin Beginning of the x iterator for the first problem
                 * @param y_it1_begin Beginning of the y iterator for the first problem
                 * @param x_it2_begin Beginning of the x iterator for the second problem
                 * @param y_it2_begin Beginning of the y iterator for the second problem
                 * @param x_it1_end End of the x iterator for the first problem
                 * @param y_it1_end End of the y iterator for the first problem
                 * @param x_it2_end End of the x iterator for the second problem
                 * @param y_it2_end End of the y iterator for the second problem
                 */
                Iterator (
                    GridProblem1D::Iterator x_it1_begin, GridProblem1D::Iterator y_it1_begin,
                    GridProblem1D::Iterator x_it2_begin, GridProblem1D::Iterator y_it2_begin,
                    GridProblem1D::Iterator x_it1_end, GridProblem1D::Iterator y_it1_end,
                    GridProblem1D::Iterator x_it2_end, GridProblem1D::Iterator y_it2_end
                ) : _x_it1(x_it1_begin), _y_it1(y_it1_begin),
                    _x_it2(x_it2_begin), _y_it2(y_it2_begin),
                    _y_it1_begin(y_it1_begin), _y_it2_begin(y_it2_begin),
                    _x_it1_end(x_it1_end), _y_it1_end(y_it1_end),
                    _x_it2_end(x_it2_end), _y_it2_end(y_it2_end)
                {}

                /**
                 * @brief Get the current solution
                */
                Zomega operator*() const {
                    if (_first_completed) {  // Solving the second problem
                        return (*_x_it2).to_Zomega() + (*_y_it2).to_Zomega() * Zomega(0, 1, 0, 0)
                            + Zomega(0, 0, 1, 0);
                    } else {  // Solving the first problem
                        return (*_x_it1).to_Zomega() + (*_y_it1).to_Zomega() * Zomega(0, 1, 0, 0);
                    }
                }

                /**
                 * @brief Increment the iterator
                 */
                void operator++() {
                    if (_first_completed) {  // Solving the second problem
                        ++_y_it2;
                        if (_y_it2 != _y_it2_end) {
                            return;
                        } else {
                            ++_x_it2;
                            _y_it2 = _y_it2_begin;
                            if (_x_it2 != _x_it2_end) {
                                return;
                            } else {
                                _second_completed = true;
                                return;
                            }
                        }
                    } else {  // Solving the first problem
                        ++_y_it1;
                        if (_y_it1 != _y_it1_end) {
                            return;
                        } else {
                            ++_x_it1;
                            _y_it1 = _y_it1_begin;
                            if (_x_it1 != _x_it1_end) {
                                return;
                            } else {
                                _first_completed = true;
                                
                                // Might be true if the second problem has no solution
                                _second_completed = !(
                                    (_x_it2 != _x_it2_end) & (_y_it2 != _y_it2_end)
                                );  
                                
                                return;
                            }
                        }
                    }
                }

                /**
                 * @brief Used to determine wether the iterator is at the end
                 * 
                 * @param _ The other iterator (unused)
                 * @return true If this iterator is at the end
                 * @return false If this iterators are not at the end
                 */
                bool operator!=(const Iterator&) const { return !_second_completed; }
        };

        /**
         * @brief Get the beginning of the iterator
         * 
         * @return Iterator The beginning of the iterator
         */
        Iterator begin() {
            return Iterator(
                _gp_x1.begin(), _gp_y1.begin(), _gp_x2.begin(), _gp_y2.begin(),
                _gp_x2.end(), _gp_y1.end(), _gp_x2.end(), _gp_y2.end()
            );
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
        Iterator end() { return begin(); }
};

#endif // GRID_ALGORITHMS_HPP

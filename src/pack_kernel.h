/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */

#ifndef PACK_KERNEL_H
#define PACK_KERNEL_H

#include "definitions.h"
#include "sycl_utils.hpp"

void clover_pack_message_left(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max,
                              clover::Buffer<double, 2> field, clover::Buffer<double, 1> left_snd, int cell_data,
                              int vertex_data, int x_face_fata, int y_face_data, int depth, int field_type,
                              int buffer_offset);
void clover_unpack_message_left(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max,
                                clover::Buffer<double, 2> field, clover::Buffer<double, 1> left_rcv, int cell_data,
                                int vertex_data, int x_face_fata, int y_face_data, int depth, int field_type,
                                int buffer_offset);
void clover_pack_message_right(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max,
                               clover::Buffer<double, 2> field, clover::Buffer<double, 1> right_snd, int cell_data,
                               int vertex_data, int x_face_fata, int y_face_data, int depth, int field_type,
                               int buffer_offset);
void clover_unpack_message_right(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max,
                                 clover::Buffer<double, 2> field, clover::Buffer<double, 1> right_rcv, int cell_data,
                                 int vertex_data, int x_face_fata, int y_face_data, int depth, int field_type,
                                 int buffer_offset);
void clover_pack_message_top(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max,
                             clover::Buffer<double, 2> field, clover::Buffer<double, 1> top_snd, int cell_data,
                             int vertex_data, int x_face_fata, int y_face_data, int depth, int field_type,
                             int buffer_offset);
void clover_unpack_message_top(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max,
                               clover::Buffer<double, 2> field, clover::Buffer<double, 1> top_rcv, int cell_data,
                               int vertex_data, int x_face_fata, int y_face_data, int depth, int field_type,
                               int buffer_offset);
void clover_pack_message_bottom(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max,
                                clover::Buffer<double, 2> field, clover::Buffer<double, 1> bottom_snd, int cell_data,
                                int vertex_data, int x_face_fata, int y_face_data, int depth, int field_type,
                                int buffer_offset);
void clover_unpack_message_bottom(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max,
                                  clover::Buffer<double, 2> field, clover::Buffer<double, 1> bottom_rcv, int cell_data,
                                  int vertex_data, int x_face_fata, int y_face_data, int depth, int field_type,
                                  int buffer_offset);

#endif

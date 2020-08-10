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


#ifndef COMMS_H
#define COMMS_H

#include "definitions.h"
#include "sycl_utils.hpp"

#include <mpi.h>

// Structure to hold MPI rank information
struct parallel_ {

	// MPI enabled?
	bool parallel;

	// Is current process the boss?
	bool boss;

	// Size of MPI communicator
	int max_task{};

	// Rank number
	int task{};

	// Constructor, (replaces clover_init_comms())
	parallel_();
};

void clover_abort();
void clover_barrier(global_variables &globals);
void clover_barrier();

std::array<int, 4> clover_decompose(const global_config &globals, parallel_ &parallel, int x_cells, int y_cells,
                                    int &left, int &right, int &bottom, int &top);
std::vector<tile_info> clover_tile_decompose(global_variables &globals, int chunk_x_cells, int chunk_y_cells);
void clover_allocate_buffers(global_variables &globals, parallel_ &parallel);

void clover_sum(double &value);
void clover_min(double &value);
void clover_allgather(double value, std::vector<double> &values);
void clover_check_error(int &error);

void clover_exchange(global_variables &globals, int fields[NUM_FIELDS], int depth);

void clover_pack_left(global_variables &globals, int tile, const int fields[NUM_FIELDS], int depth,
                      int left_right_offset[NUM_FIELDS]);
void
clover_send_recv_message_left(global_variables &globals, clover::Buffer<double, 1> &left_snd_buffer,
                              clover::Buffer<double, 1> &left_rcv_buffer, int total_size, int tag_send,
                              int tag_recv, MPI_Request &req_send, MPI_Request &req_recv);
void clover_unpack_left(global_variables &globals, const int fields[NUM_FIELDS], int tile, int depth,
                        int left_right_offset[NUM_FIELDS]);

void clover_pack_right(global_variables &globals, int tile, const int fields[NUM_FIELDS], int depth,
                       int left_right_offset[NUM_FIELDS]);
void
clover_send_recv_message_right(global_variables &globals, clover::Buffer<double, 1> &right_snd_buffer,
                               clover::Buffer<double, 1> &right_rcv_buffer, int total_size,
                               int tag_send, int tag_recv, MPI_Request &req_send,
                               MPI_Request &req_recv);
void clover_unpack_right(global_variables &globals, const int fields[NUM_FIELDS], int tile, int depth,
                         int left_right_offset[NUM_FIELDS]);

void clover_pack_top(global_variables &globals, int tile, const int fields[NUM_FIELDS], int depth,
                     int bottom_top_offset[NUM_FIELDS]);
void clover_send_recv_message_top(global_variables &globals, clover::Buffer<double, 1> &top_snd_buffer,
                                  clover::Buffer<double, 1> &top_rcv_buffer, int total_size,
                                  int tag_send, int tag_recv, MPI_Request &req_send,
                                  MPI_Request &req_recv);
void clover_unpack_top(global_variables &globals, const int fields[NUM_FIELDS], int tile, int depth,
                       int bottom_top_offset[NUM_FIELDS]);

void clover_pack_bottom(global_variables &globals, int tile, const int fields[NUM_FIELDS], int depth,
                        int bottom_top_offset[NUM_FIELDS]);
void clover_send_recv_message_bottom(global_variables &globals,
                                     clover::Buffer<double, 1> &bottom_snd_buffer,
                                     clover::Buffer<double, 1> &top_rcv_buffer, int total_size,
                                     int tag_send, int tag_recv, MPI_Request &req_send,
                                     MPI_Request &req_recv);
void clover_unpack_bottom(global_variables &globals, const int fields[NUM_FIELDS], int tile, int depth,
                          int bottom_top_offset[NUM_FIELDS]);
#endif


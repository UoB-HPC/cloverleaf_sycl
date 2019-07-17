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


//  @brief Fortran mpi buffer packing kernel
//  @author Wayne Gaudin
//  @details Packs/unpacks mpi send and receive buffers

#include "pack_kernel.h"
#include "sycl_utils.hpp"

void clover_pack_message_left(handler &h, int x_min, int x_max, int y_min, int y_max,
                              Accessor<double, 2, RW>::Type field,
                              Accessor<double, 1, RW>::Type left_snd_buffer,
                              int cell_data, int vertex_data, int x_face_data, int y_face_data,
                              int depth, int field_type, int buffer_offset) {

	// Pack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	// DO k=y_min-depth,y_max+y_inc+depth

	par_ranged<class APPEND_LN(clover_pack_message_left)>(
			h, {y_min - depth + 1, y_max + y_inc + depth + 2}, [=](id<1> k) {
				for (int j = 0; j < depth; ++j) {
					int index = buffer_offset + j + (k.get(0) + depth - 1) * depth;
					left_snd_buffer[index] = field[x_min + x_inc - 1 + j][k.get(0)];
				}
			});

}


void clover_unpack_message_left(handler &h, int x_min, int x_max, int y_min, int y_max,
                                Accessor<double, 2, RW>::Type field,
                                Accessor<double, 1, RW>::Type left_rcv_buffer,
                                int cell_data, int vertex_data, int x_face_data, int y_face_data,
                                int depth, int field_type, int buffer_offset) {

	// Upnack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	// DO k=y_min-depth,y_max+y_inc+depth

	par_ranged<class APPEND_LN(clover_unpack_message_left)>(
			h, {y_min - depth + 1, y_max + y_inc + depth + 2}, [=](id<1> k) {
				for (int j = 0; j < depth; ++j) {
					int index = buffer_offset + j + (k.get(0) + depth - 1) * depth;
					field[x_min - j][k.get(0)] = left_rcv_buffer[index];
				}
			});

}


void clover_pack_message_right(handler &h, int x_min, int x_max, int y_min, int y_max,
                               Accessor<double, 2, RW>::Type field,
                               Accessor<double, 1, RW>::Type right_snd_buffer,
                               int cell_data, int vertex_data, int x_face_data, int y_face_data,
                               int depth, int field_type, int buffer_offset) {

	// Pack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	// DO k=y_min-depth,y_max+y_inc+depth

	par_ranged<class APPEND_LN(clover_pack_message_right)>(
			h, {y_min - depth + 1, y_max + y_inc + depth + 2}, [=](id<1> k) {
				for (int j = 0; j < depth; ++j) {
					int index = buffer_offset + j + (k.get(0) + depth - 1) * depth;
					right_snd_buffer[index] = field[x_min + 1 + j][k.get(0)];
				}
			});

}


void clover_unpack_message_right(handler &h, int x_min, int x_max, int y_min, int y_max,
                                 Accessor<double, 2, RW>::Type field,
                                 Accessor<double, 1, RW>::Type right_rcv_buffer,
                                 int cell_data, int vertex_data, int x_face_data, int y_face_data,
                                 int depth, int field_type, int buffer_offset) {

	// Upnack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	// DO k=y_min-depth,y_max+y_inc+depth

	par_ranged<class APPEND_LN(clover_pack_message_left)>(
			h, {y_min - depth + 1, y_max + y_inc + depth + 2}, [=](id<1> k) {
				for (int j = 0; j < depth; ++j) {
					int index = buffer_offset + j + (k.get(0) + depth - 1) * depth;
					right_rcv_buffer[index] = field[x_max + x_inc + j][k.get(0)];
				}
			});

}

void clover_pack_message_top(handler &h, int x_min, int x_max, int y_min, int y_max,
                             Accessor<double, 2, RW>::Type field, Accessor<double, 1, RW>::Type top_snd_buffer,
                             int cell_data, int vertex_data, int x_face_data, int y_face_data,
                             int depth, int field_type, int buffer_offset) {

	// Pack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	for (int k = 0; k < depth; ++k) {
		// DO j=x_min-depth,x_max+x_inc+depth

		par_ranged<class APPEND_LN(clover_pack_message_top)>(
				h, {x_min - depth + 1, x_max + x_inc + depth + 2}, [=](id<1> j) {
					int index = buffer_offset + k + (j.get(0) + depth - 1) * depth;
					top_snd_buffer[index] = field[j.get(0)][y_max + 1 - k];
				});
	}
}

void clover_unpack_message_top(handler &h, int x_min, int x_max, int y_min, int y_max,
                               Accessor<double, 2, RW>::Type field,
                               Accessor<double, 1, RW>::Type top_rcv_buffer,
                               int cell_data, int vertex_data, int x_face_data, int y_face_data,
                               int depth, int field_type, int buffer_offset) {

	// Unpack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	for (int k = 0; k < depth; ++k) {
		// DO j=x_min-depth,x_max+x_inc+depth

		par_ranged<class APPEND_LN(clover_unpack_message_top)>(
				h, {x_min - depth + 1, x_max + x_inc + depth + 2}, [=](id<1> j) {
					int index = buffer_offset + k + (j.get(0) + depth - 1) * depth;
					field[j.get(0)][y_max + y_inc + k] = top_rcv_buffer[index];
				});
	}
}


void clover_pack_message_bottom(handler &h, int x_min, int x_max, int y_min, int y_max,
                                Accessor<double, 2, RW>::Type field,
                                Accessor<double, 1, RW>::Type bottom_snd_buffer,
                                int cell_data, int vertex_data, int x_face_data, int y_face_data,
                                int depth, int field_type, int buffer_offset) {

	// Pack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	for (int k = 0; k < depth; ++k) {
		// DO j=x_min-depth,x_max+x_inc+depth

		par_ranged<class APPEND_LN(clover_pack_message_bottom)>(
				h, {x_min - depth + 1, x_max + x_inc + depth + 2}, [=](id<1> j) {
					int index = buffer_offset + k + (j.get(0) + depth - 1) * depth;
					bottom_snd_buffer[index] = field[j.get(0)][y_min + y_inc - 1 + k];
				});
	}
}

void clover_unpack_message_bottom(handler &h, int x_min, int x_max, int y_min, int y_max,
                                  Accessor<double, 2, RW>::Type field,
                                  Accessor<double, 1, RW>::Type bottom_rcv_buffer,
                                  int cell_data, int vertex_data, int x_face_data, int y_face_data,
                                  int depth, int field_type, int buffer_offset) {

	// Unpack

	int x_inc, y_inc;

	// These array modifications still need to be added on, plus the donor data location changes as in update_halo
	if (field_type == cell_data) {
		x_inc = 0;
		y_inc = 0;
	}
	if (field_type == vertex_data) {
		x_inc = 1;
		y_inc = 1;
	}
	if (field_type == x_face_data) {
		x_inc = 1;
		y_inc = 0;
	}
	if (field_type == y_face_data) {
		x_inc = 0;
		y_inc = 1;
	}

	for (int k = 0; k < depth; ++k) {
		// DO j=x_min-depth,x_max+x_inc+depth

		par_ranged<class APPEND_LN(clover_unpack_message_top)>(
				h, {x_min - depth + 1, x_max + x_inc + depth + 2}, [=](id<1> j) {
					int index = buffer_offset + k + (j.get(0) + depth - 1) * depth;
					field[j.get(0)][y_min - k] = bottom_rcv_buffer[index];
				});
	}
}

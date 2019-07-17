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

void
clover_pack_message_left(handler&h, int x_min, int x_max, int y_min, int y_max, AccDP2RW::Type field,
                         AccDP1RW::Type left_snd_buffer, int cell_data, int vertex_data,
                         int x_face_fata, int y_face_data, int depth, int field_type,
                         int buffer_offset);
void clover_unpack_message_left(handler&h, int x_min, int x_max, int y_min, int y_max,
                                AccDP2RW::Type field,
                                AccDP1RW::Type left_rcv_buffer, int cell_data,
                                int vertex_data, int x_face_fata, int y_face_data, int depth,
                                int field_type, int buffer_offset);
void clover_pack_message_right(handler&h, int x_min, int x_max, int y_min, int y_max,
                               AccDP2RW::Type field,
                               AccDP1RW::Type right_snd_buffer, int cell_data,
                               int vertex_data, int x_face_fata, int y_face_data, int depth,
                               int field_type, int buffer_offset);
void clover_unpack_message_right(handler&h, int x_min, int x_max, int y_min, int y_max,
                                 AccDP2RW::Type field,
                                 AccDP1RW::Type right_rcv_buffer, int cell_data,
                                 int vertex_data, int x_face_fata, int y_face_data, int depth,
                                 int field_type, int buffer_offset);
void
clover_pack_message_top(handler&h, int x_min, int x_max, int y_min, int y_max, AccDP2RW::Type field,
                        AccDP1RW::Type top_snd_buffer, int cell_data, int vertex_data,
                        int x_face_fata, int y_face_data, int depth, int field_type,
                        int buffer_offset);
void clover_unpack_message_top(handler&h, int x_min, int x_max, int y_min, int y_max,
                               AccDP2RW::Type field,
                               AccDP1RW::Type top_rcv_buffer, int cell_data,
                               int vertex_data, int x_face_fata, int y_face_data, int depth,
                               int field_type, int buffer_offset);
void clover_pack_message_bottom(handler&h, int x_min, int x_max, int y_min, int y_max,
                                AccDP2RW::Type field,
                                AccDP1RW::Type bottom_snd_buffer, int cell_data,
                                int vertex_data, int x_face_fata, int y_face_data, int depth,
                                int field_type, int buffer_offset);
void clover_unpack_message_bottom(handler&h, int x_min, int x_max, int y_min, int y_max,
                                  AccDP2RW::Type field,
                                  AccDP1RW::Type bottom_rcv_buffer, int cell_data,
                                  int vertex_data, int x_face_fata, int y_face_data, int depth,
                                  int field_type, int buffer_offset);

#endif


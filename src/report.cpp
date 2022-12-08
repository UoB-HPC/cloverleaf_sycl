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

//  @brief Controls error reporting
//  @author Wayne Gaudin
//  @details Outputs error messages and aborts the calculation.

#include "report.h"

#include "comms.h"

#include <iostream>

extern std::ostream g_out;

void report_error(char *location, char *error) {

  std::cout << std::endl
            << "Error from " << location << ":" << std::endl
            << error << std::endl
            << "CLOVER is terminating." << std::endl
            << std::endl;

  g_out << std::endl
        << "Error from " << location << ":" << std::endl
        << error << std::endl
        << "CLOVER is terminating." << std::endl
        << std::endl;

  clover_abort();
}

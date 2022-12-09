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

//  @brief Top level initialisation routine
//  @author Wayne Gaudin
//  @details Checks for the user input and either invokes the input reader or
//  switches to the internal test problem. It processes the input and strips
//  comments before writing a final input file.
//  It then calls the start routine.

#include "initialise.h"
#include "read_input.h"
#include "report.h"
#include "start.h"
#include "version.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

extern std::ostream g_out;
std::ofstream of;

struct RunConfig {
  std::string file;
  cl::sycl::device device;
};

std::string deviceName(sycl::info::device_type type) {
  //@formatter:off
  switch (type) {
    case sycl::info::device_type::cpu: return "cpu";
    case sycl::info::device_type::gpu: return "gpu";
    case sycl::info::device_type::accelerator: return "accelerator";
    case sycl::info::device_type::custom: return "custom";
    case sycl::info::device_type::automatic: return "automatic";
    case sycl::info::device_type::host: return "host";
    case sycl::info::device_type::all: return "all";
    default: return "(unknown: " + std::to_string(static_cast<unsigned int>(type)) + ")";
  }
  //@formatter:on
}

// dumps device info to stdout
void printDetailed(const cl::sycl::device &device, size_t index) {
  auto exts = device.get_info<cl::sycl::info::device::extensions>();
  std::ostringstream extensions;
  std::copy(exts.begin(), exts.end(), std::ostream_iterator<std::string>(extensions, ","));

  auto type = device.get_info<cl::sycl::info::device::device_type>();
  cl::sycl::platform platform = device.get_platform();
  std::cout << " + Device        : " << device.get_info<cl::sycl::info::device::name>() << "\n";
  std::cout << "   - Index      : " << index << "\n";
  std::cout << "   - Type       : " << deviceName(type) << "\n";
  std::cout << "   - Vendor     : " << device.get_info<cl::sycl::info::device::vendor>() << "\n";
  std::cout << "   - Extensions : " << extensions.str() << "\n";
  std::cout << "   + Platform   : " << platform.get_info<cl::sycl::info::platform::name>() << "\n";
  std::cout << "      - Vendor  : " << platform.get_info<cl::sycl::info::platform::vendor>() << "\n";
  std::cout << "      - Version : " << platform.get_info<cl::sycl::info::platform::version>() << "\n";
  std::cout << "      - Profile : " << platform.get_info<cl::sycl::info::platform::profile>() << "\n";
}

void printSimple(const cl::sycl::device &device, size_t index) {
  std::cout << std::setw(3) << index << ". " << device.get_info<cl::sycl::info::device::name>() << "("
            << deviceName(device.get_info<cl::sycl::info::device::device_type>()) << ")" << std::endl;
}

void printHelp(const std::string &name) {
  std::cout << std::endl;
  std::cout << "Usage: " << name << " [OPTIONS]\n\n"
            << "Options:\n"
            << "  -h  --help               Print the message\n"
            << "      --list               List available devices\n"
            << "      --list-detailed      List available devices and capabilities\n"
            << "      --device <INDEX>     Select device at INDEX from output of --list\n"
            << "      --file               Custom clover.in file (defaults to clover.in if unspecified)\n"
            << std::endl;
}

RunConfig parseArgs(const std::vector<cl::sycl::device> &devices, const std::vector<std::string> &args) {

  const auto readParam = [&args](size_t current, const std::string &emptyMessage, auto map) {
    if (current + 1 < args.size()) {
      return map(args[current + 1]);
    } else {
      std::cerr << emptyMessage << std::endl;
      printHelp(args[0]);
      std::exit(EXIT_FAILURE);
    }
  };

  auto config = RunConfig{"clover.in", devices[0]};
  for (size_t i = 0; i < args.size(); ++i) {
    const auto arg = args[i];

    if (arg == "--help" || arg == "-h") {
      printHelp(args[0]);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--list-detailed") {
      for (size_t j = 0; j < devices.size(); ++j)
        printDetailed(devices[j], j);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--list") {
      for (size_t j = 0; j < devices.size(); ++j)
        printSimple(devices[j], j);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--device") {
      readParam(i, "--device specified but no size was given", [&config](const auto &param) {
        auto devices = cl::sycl::device::get_devices();

        try {
          config.device = devices.at(std::stoul(param));
        } catch (const std::exception &e) {
          std::cout << "Unable to parse/select device index `" << param << "`:" << e.what() << std::endl;
          std::cout << "Attempting to match device with substring  `" << param << "`" << std::endl;

          auto matching = std::find_if(devices.begin(), devices.end(), [param](const cl::sycl::device &device) {
            return device.get_info<cl::sycl::info::device::name>().find(param) != std::string::npos;
          });
          if (matching != devices.end()) {
            config.device = *matching;
            std::cout << "Using first device matching substring `" << param << "`" << std::endl;
          } else if (devices.size() == 1)
            std::cerr << "No matching device but there's only one device, will be using that anyway" << std::endl;
          else {
            std::cerr << "No matching devices" << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      });
    } else if (arg == "--file") {
      readParam(i, "--file specified but no file was given", [&config](const auto &param) { config.file = param; });
    }
  }
  return config;
}

std::unique_ptr<global_variables> initialise(parallel_ &parallel, const std::vector<std::string> &args) {

  global_config config;

  if (parallel.boss) {
    of.open("clover.out");
    if (!of.is_open()) report_error((char *)"initialise", (char *)"Error opening clover.out file.");

    g_out.rdbuf(of.rdbuf());

  } else {
    g_out.rdbuf(std::cout.rdbuf());
  }

  if (parallel.boss) {
    g_out << "Clover Version " << g_version << std::endl
          << "Kokkos Version" << std::endl
          << "Task Count " << parallel.max_task << std::endl
          << std::endl;

    std::cout << "Output file clover.out opened. All output will go there." << std::endl;
  }

  clover_barrier();

  const auto &devices = cl::sycl::device::get_devices();
  if (devices.empty()) {
    std::cerr << "No SYCL devices available" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto runConfig = parseArgs(devices, args);
  auto file = runConfig.file;
  auto selectedDevice = runConfig.device;

  std::cout << "Detected SYCL devices:" << std::endl;
  for (size_t i = 0; i < devices.size(); ++i)
    printSimple(devices[i], i);

  std::cout << "Using SYCL device: " << std::endl;
  std::cout << "Device    : " << selectedDevice.get_info<cl::sycl::info::device::name>() << std::endl;
  std::cout << "\tType    : " << deviceName(selectedDevice.get_info<cl::sycl::info::device::device_type>())
            << std::endl;
  std::cout << "\tProfile : " << selectedDevice.get_info<cl::sycl::info::device::profile>() << std::endl;
  std::cout << "\tVersion : " << selectedDevice.get_info<cl::sycl::info::device::version>() << std::endl;
  std::cout << "\tVendor  : " << selectedDevice.get_info<cl::sycl::info::device::vendor>() << std::endl;
  std::cout << "\tDriver  : " << selectedDevice.get_info<cl::sycl::info::device::driver_version>() << std::endl;

  std::ifstream g_in;
  g_out << "Clover will run from the following input:-" << std::endl << std::endl;

  if (!args.empty()) {
    std::cout << "Args:";
    for (const auto &arg : args)
      std::cout << " " << arg;
    std::cout << std::endl;
  }

  if (!file.empty()) {
    std::cout << "Using input: `" << file << "`" << std::endl;
    g_in.open(file);
    if (g_in.fail()) {
      std::cerr << "Unable to open file: `" << file << "`" << std::endl;
      std::exit(1);
    }
  } else {
    std::cout << "No input file specified, using default input" << std::endl;
    std::ofstream out_unit("clover.in");
    out_unit << "*clover" << std::endl
             << " state 1 density=0.2 energy=1.0" << std::endl
             << " state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0" << std::endl
             << " x_cells=10" << std::endl
             << " y_cells=2" << std::endl
             << " xmin=0.0" << std::endl
             << " ymin=0.0" << std::endl
             << " xmax=10.0" << std::endl
             << " ymax=2.0" << std::endl
             << " initial_timestep=0.04" << std::endl
             << " timestep_rise=1.5" << std::endl
             << " max_timestep=0.04" << std::endl
             << " end_time=3.0" << std::endl
             << " test_problem 1" << std::endl
             << "*endclover" << std::endl;
    out_unit.close();
    g_in.open("clover.in");
  }

  clover_barrier();

  if (parallel.boss) {
    g_out << std::endl << "Initialising and generating" << std::endl << std::endl;
  }

  read_input(g_in, parallel, config);

  clover_barrier();

  //	globals.step = 0;
  config.number_of_chunks = parallel.max_task;

  auto globals = start(parallel, config, selectedDevice);

  clover_barrier(*globals);

  if (parallel.boss) {
    g_out << "Starting the calculation" << std::endl;
  }

  g_in.close();

  return globals;
}

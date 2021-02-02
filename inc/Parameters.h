#pragma once

#include <Kokkos_Core.hpp>
#include <bits/stdc++.h>

#include "../external/inih/INIReader.h"
#include "DataTypes.h"

namespace kokkos_virtual {

/**
 * Simple structure holding the parameters of the run
 **/
struct Parameters {
  /**
   * Constructor
   * 
   * @param filename the filename corresponding to the .ini file to read
   **/
  Parameters(std::string filename) {
    // The actual reader
    INIReader reader(filename);

    // Constants
    G             = reader.GetReal("run", "G",    6.67e-11);
    dt            = reader.GetReal("run", "dt",   1.0);
    tmax          = reader.GetReal("run", "tmax", 1.0); 
    output_prefix = reader.Get("run", "output_prefix", "run");
    output_freq   = reader.GetInteger("run", "output_freq", 10);
    update_type   = reader.Get("run", "update_type", "euler");
    nb_teams      = reader.GetInteger("run", "nb_teams", 128);
    
    // Extracting the path
    std::string path = filename;
    while (!path.empty() && path.back() != '/')
      path = path.substr(0, path.size()-1);

    // And extracting the filename
    input_filename = path + reader.Get("run", "input_filename", "");

    // Error checking
    if (input_filename == "") {
      std::cerr << "ERROR: No input file provided !" << std::endl;
      std::exit(2);
    }
  }

  /** 
   * Prints the parameters of the run
   **/
  void PrintParameters() {
    std::cout << " ===================== Run Parameters ================== " << std::endl;
    std::cout << " . Gravitational constant     G    = " << G    << std::endl;
    std::cout << " . Maximum time of simulation tmax = " << tmax << std::endl;
    std::cout << " . Time-step                  dt   = " << dt   << std::endl;
    std::cout << " . Output frequency           fout = " << output_freq << std::endl;
    std::cout << " . Update type                     = " << update_type << std::endl;
    std::cout << " ======================================================= " << std::endl;
  }

  // Parameter list
  real_t      G;              // !< Gravitational constant to be used
  real_t      tmax;           // !< Maximum time of simulation
  real_t      dt;             // !< Time-step
  std::string input_filename; // !< Filename corresponding to the initial condition
  std::string output_prefix;  // !< Prefix added to each output filename
  uint        output_freq;    // !< Frequency of outputs in iterations
  std::string update_type;    // !< Type of solver for the update
  uint        nb_teams;       // !< Size of a thread-league
};

}
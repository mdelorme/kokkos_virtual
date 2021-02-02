#pragma once

#include <bits/stdc++.h>
#include <Kokkos_Core.hpp>

#include "Parameters.h"
#include "DataTypes.h"

namespace kokkos_virtual {
/**
 * @class DataManager
 * @brief A class made to load and save data
 * 
 * Data is loaded from an ascii description using 7 columns. x, y, z, vx, vy, vz, mass
 * Data is saved to a vtk unstructured grid.
 **/
class DataManager {
 public:
  DataManager() {};

  /**
   * Initializes the data from the input file provided
   * 
   * @param filename the name of the file to load particles from
   * @return the particle set on device. Note that the host version is NOT given since it
   *         should only be used for IOs and nothign else !
   * 
   * @note This method loads data under the form of a simple ascii list 
   *       with 7 columns, 3 positions, 3 velocities and a mass. The data
   *       has units consistent with the value of G provided in the .ini file
   * 
   * @note The data is allocated inside this function. No need to allocate particles and
   *       particles_h before that.
   * 
   * @todo Particle copy should be done on parallel on host (openmp)
   **/
  ParticleArray InitData(std::string filename) {
    std::cout << " == Initializing particles from file " << filename << std::endl;

    // We are reading the data from the file using only one process
    // This could be done in parallel ... But not adapted to this format.
    using Particle = std::array<real_t, NB_STORED_FIELDS>;
    std::vector<Particle> pdata;

    // Reading data from the file
    std::ifstream f_in;
    f_in.open(filename);
    if (!f_in.good()) {
      std::cerr << "ERROR: File " << filename << " could not be opened !" << std::endl;
      std::exit(3);
    }

    while (!f_in.eof() && f_in.good()) {
      Particle p;
      f_in >> p[IX] >> p[IY] >> p[IZ] >> p[IVX] >> p[IVY] >> p[IVZ] >> p[IM];
      pdata.push_back(p);
    }

    // Initializing the particle array and mirroring
    const uint nParticles = pdata.size();
    particles = ParticleArray("Particles", NB_FIELDS, nParticles);
    particles_h = Kokkos::create_mirror(particles);

    // Copying the data on host
    // TODO : This should be done in parallel !
    for (uint iPart=0; iPart < nParticles; ++iPart) {
      particles_h(IX,  iPart) = pdata[iPart][IX];
      particles_h(IY,  iPart) = pdata[iPart][IY];
      particles_h(IZ,  iPart) = pdata[iPart][IZ];
      particles_h(IVX, iPart) = pdata[iPart][IVX];
      particles_h(IVY, iPart) = pdata[iPart][IVY];
      particles_h(IVZ, iPart) = pdata[iPart][IVZ];
      particles_h(IM,  iPart) = pdata[iPart][IM];
    }
     
    Kokkos::deep_copy(particles, particles_h);
    std::cout << "   -> Done. Read " << nParticles << " particles." << std::endl;

    return particles;
  }

  /**
   * Constructs a string corresponding to the current output
   * 
   * @param params the parameters of the run
   * @param iteration the current iteration
   * @result the string corresponding to the output filename for the given iteration
   **/
  std::string BuildFilename(Parameters &params, uint iteration) {
    std::ostringstream oss;
    oss << params.output_prefix << "_" << std::setw(8) << std::setfill('0') << iteration << ".vtu";
    return oss.str();
  }

  /**
   * Saves the data to an unstructured ascii vtk file
   * 
   * @param filename the filename to save the data to
   **/ 
  void SaveData(std::string filename) {
    // Bringing back the data from the device
    Kokkos::deep_copy(particles_h, particles);

    std::ofstream f_out;
    f_out.open(filename);

    if (!f_out.good()) {
      std::cerr << "ERROR : Impossible to write to file " << filename << std::endl;
      std::exit(4); // Ok, maybe we should exit in a less brutal way. Todo...
    }

    // Yes this is ugly, and should be written using the vtk library ...
    uint nbParticles = particles.extent(1);
    f_out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    f_out << "  <UnstructuredGrid>" << std::endl;
    f_out << "    <Piece NumberOfPoints=\"" << nbParticles << "\" NumberOfCells=\"0\">" << std::endl;
    f_out << "      <Points>" << std::endl;
    f_out << "        <DataArray Name=\"Position\" type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    for (uint i=0; i < nbParticles; ++i)
      f_out << particles_h(IX, i) << " " << particles_h(IY, i) << " " << particles_h(IZ, i) << " ";
    f_out << std::endl;
    f_out << "        </DataArray>" << std::endl;
    f_out << "      </Points>" << std::endl;
    f_out << "      <PointData Scalars=\"Mass\" Vectors=\"Velocity\">" << std::endl;
    f_out << "        <DataArray Name=\"Velocity\" type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    for (uint i=0; i < nbParticles; ++i)
      f_out << particles_h(IVX, i) << " " << particles_h(IVY, i) << " " << particles_h(IVZ, i) << " ";
    f_out << std::endl;      
    f_out << "        </DataArray>" << std::endl;
    f_out << "        <DataArray Name=\"Mass\" type=\"Float32\" format=\"ascii\">" << std::endl;
    for (uint i=0; i < nbParticles; ++i)
      f_out << particles_h(IM, i) << " ";
    f_out << std::endl;      
    f_out << "        </DataArray>" << std::endl;
    f_out << "      </PointData>" << std::endl;
    f_out << "      <CellData></CellData>" << std::endl;
    f_out << "      <Cells>" << std::endl;
    f_out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    f_out << "        </DataArray>" << std::endl;
    f_out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    f_out << "        </DataArray>" << std::endl;
    f_out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << std::endl;
    f_out << "        </DataArray>" << std::endl;
    f_out << "      </Cells>" << std::endl;
    f_out << "    </Piece>" << std::endl;
    f_out << "  </UnstructuredGrid>" << std::endl;
    f_out << "</VTKFile>" << std::endl;

    f_out.close();
  }


  ParticleArray     particles;   //!< Particles on device
  ParticleArrayHost particles_h; //!< Particles on host
};

}
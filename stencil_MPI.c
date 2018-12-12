
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mm_malloc.h>
#include "mpi.h"

#define MASTER 0

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil_mid(const int nx, const int ny, float *  image, float *  tmp_image);
void stencil_master(const int nx, const int ny, float *  image, float *  tmp_image);
void stencil_last(const int nx, const int ny, float *  image, float *  tmp_image);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
int calc_ncols_from_rank(int rank, int size, int ny);
double wtime(void);

int main(int argc, char *argv[]) {
  int rank;              // Rank indentifier
  int size;              // Size of cohort
  int left;              /* the rank of the process to the left */
  int right;             /* the rank of the process to the right */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */
  float *localimage;     /* local image grid */
  float *tmp_localimage; /* local tmp image grid */
  int i;                 // iterator
  int j;                 // iterator
  MPI_Status status;     /* struct used by MPI_Recv */

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  /* MPI_Init returns once it has started up processes */
  /* get size and rank */ 
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  /* 
  ** determine process ranks to the left and right of rank
  ** respecting periodic boundary conditions
  */
  left = (rank == 0) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  /* 
  ** determine local grid size
  ** each rank gets all the rows, but a subset of the number of columns
  */
  local_nrows = ny;
  local_ncols = calc_ncols_from_rank(rank, size, ny);
  if (local_ncols < 1) 
  {
    fprintf(stderr,"Error: too many processes:- local_ncols < 1\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  /*
  ** allocate space for:
  ** - the local grid with 2 extra columns added for the halos. In case of first column and last column, only 1 halo
  */
  localimage = (float*)malloc(sizeof(float*) * local_nrows * (local_ncols + 2));
  tmp_localimage = (float*)malloc(sizeof(float*) * local_nrows * (local_ncols + 2));
  
  // Allocate the image globally
  float *image = _mm_malloc(sizeof(float)*nx*ny, 32);
  float *tmp_image = _mm_malloc(sizeof(float)*nx*ny, 32);
  float *result = _mm_malloc(sizeof(float)*nx*ny, 32);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  /*
  ** allocate space for:
  ** - the local grid with 2 extra columns added for the halos. In case of first column and last column, only 1 halo
  */
  localimage = (float*)malloc(sizeof(float*) * local_nrows * (local_ncols + 2));
  tmp_localimage = (float*)malloc(sizeof(float*) * local_nrows * (local_ncols + 2));

  /*
  ** initialize the local grid
  */
  for(j=1; j<local_ncols+1; ++j)
  {
    for(i=0;i<local_nrows; ++i) 
    {
      localimage [i+j*ny] = image [i+((rank*(nx/size)-1)+j)*ny];
      tmp_localimage [i+j*ny] = tmp_image[i+((rank*(nx/size)-1)+j)*ny];
    }
  }

  // SEND HALO LEFT FIRST
  if (rank != 0)
  {
    MPI_Ssend(&localimage[1*ny], ny, MPI_FLOAT, left, 0, MPI_COMM_WORLD);
  }
  if (rank != size-1)
  {
    MPI_Recv(&localimage[(local_ncols+1)*ny], ny, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);
    printf("RECEIVED! %d", rank);
  }
  // THEN SEND RIGHT
  if (rank != size-1)
  {
    MPI_Ssend(&localimage[(local_ncols)*ny], ny, MPI_FLOAT, right, 0, MPI_COMM_WORLD);
  }
  if (rank != 0)
  {
    MPI_Recv(&localimage[0], ny, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);
    printf("RECEIVED! %d", rank);
  }

  if (rank == MASTER)
  {
    output_image("CHECK.pgm", local_ncols +2, ny, localimage);
    free(image);
  }

  /*-----------------------------------------------------------START STENCIL-----------------------------------------------------------*/
  double tic = wtime();
  for (int t = 0; t < niters; ++t)
  {
    //CARRY OUT STENCIL
    if (rank == MASTER)
    {
      stencil_master(local_ncols, ny, localimage, tmp_localimage);
    }
    else if (rank == size - 1)
    {
      stencil_last(local_ncols, ny, localimage, tmp_localimage);
    }
    else
    {
      stencil_mid(local_ncols, ny, localimage, tmp_localimage);
    }
    // SEND HALO LEFT FIRST
    if (rank != 0)
    {
      MPI_Ssend(&tmp_localimage[1*ny], ny, MPI_FLOAT, left, 0, MPI_COMM_WORLD);
    }
    if (rank != size-1)
    {
      MPI_Recv(&tmp_localimage[(local_ncols+1)*ny], ny, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);
    }
    // THEN SEND RIGHT
    if (rank != size-1)
    {
      MPI_Ssend(&tmp_localimage[(local_ncols)*ny], ny, MPI_FLOAT, right, 0, MPI_COMM_WORLD);
    }
    if (rank != 0)
    {
      MPI_Recv(&tmp_localimage[0], ny, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);
    }



    //CARRY OUT STENCIL AGAIN
    if (rank == MASTER)
    {
      stencil_master(local_ncols, ny, tmp_localimage, localimage);
    }
    else if (rank == size - 1)
    {
      stencil_last(local_ncols, ny, tmp_localimage, localimage);
    }
    else
    {
      stencil_mid(local_ncols, ny, tmp_localimage, localimage);
    }
    // SEND HALO LEFT FIRST
    if (rank != 0)
    {
      MPI_Ssend(&localimage[1*ny], ny, MPI_FLOAT, left, 0, MPI_COMM_WORLD);
    }
    if (rank != size-1)
    {
      MPI_Recv(&localimage[(local_ncols+1)*ny], ny, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &status);
    }
    // THEN SEND RIGHT
    if (rank != size-1)
    {
      MPI_Ssend(&localimage[(local_ncols)*ny], ny, MPI_FLOAT, right, 0, MPI_COMM_WORLD);
    }
    if (rank != 0)
    {
      MPI_Recv(&localimage[0], ny, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);
    }
  }
  double toc = wtime();

  /*-----------------------------------------------------------END STENCIL-----------------------------------------------------------*/

  if (rank == 0) 
  {
    for(j=1; j<local_ncols+1; ++j)
    {
      for(i=0;i<local_nrows; ++i) 
      {
        result[i+(j-1)*ny] = localimage [i+j*ny];
      }
    }
    for (int r = 1; r<size; ++r)
    {
      int cols = calc_ncols_from_rank(r, size, ny);
      for(j=1; j<cols+1; ++j)
      {
        MPI_Recv(&result[((r*(nx/size)-1)+j)*ny], ny, MPI_FLOAT, r, 0, MPI_COMM_WORLD, &status);
      }
    }
  }
  else
  {
    for(j=1; j<local_ncols+1; ++j)
      {
        MPI_Ssend(&localimage[j*ny], ny, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      }
  }

  if (rank == 0) 
  {
    printf("REACHES HERE, RANK : %d\n", rank);
  }

  if (rank == MASTER)
  {
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");
    printf("With %d processes\n", size);

    output_image(OUTPUT_FILE, nx, ny, result);
    free(image);
  }

  /* don't forget to tidy up when we're done */
  MPI_Finalize();

  /* and exit the program */
  return EXIT_SUCCESS;
}

void stencil_mid(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  // first row
  for (int i = 1; i < nx+1; ++i) {
    tmp_image[i*ny] = image[i*ny] * 0.6 + (image[(i*ny)+1] + image[(i*ny)+ny] + image[(i*ny)-ny]) * 0.1;
  }

  // last row
  for (int i = 1; i < nx+1; ++i) {
    tmp_image[i*ny+(ny-1)] = image[i*ny+(ny-1)] * 0.6 + (image[i*ny+(ny-1)-1] + image[i*ny+(ny-1)-ny] + image[i*ny+(ny-1)+ny]) * 0.1;
  }

  // middle boxes
  #pragma vector aligned
  for (int i = 1; i < nx+1; ++i) {
    for (int j = 1; j < ny-1; ++j) {
      tmp_image[j+i*ny] = image[j+i*ny] * 0.6 + (image[j+i*ny-ny] + image[j+i*ny+ny] + image[j+i*ny-1] + image[j+i*ny+1]) * 0.1;
    }
  }
}

void stencil_master(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  // Case first column first row
  tmp_image[0+ny] = image[0+ny] * 0.6 + (image[1+ny] + image[2*ny]) * 0.1;

  // Case first column last row
  tmp_image[2*ny-1] = image[2*ny-1] * 0.6 + (image[2*ny-2] + image[(2*ny-1)+ny]) * 0.1;

  // first column
  for (int i = 1; i < ny-1; ++i) {
    tmp_image[i+ny] = image[i+ny] * 0.6 + (image[i-1+ny] + image[i+1+ny] + image[i+2*ny]) * 0.1;
  }

  // first row
  for (int i = 2; i < nx+1; ++i) {
    tmp_image[i*ny] = image[i*ny] * 0.6 + (image[(i*ny)+1] + image[(i*ny)+ny] + image[(i*ny)-ny]) * 0.1;
  }

  // last row
  for (int i = 2; i < nx+1; ++i) {
    tmp_image[i*ny+(ny-1)] = image[i*ny+(ny-1)] * 0.6 + (image[i*ny+(ny-1)-1] + image[i*ny+(ny-1)-ny] + image[i*ny+(ny-1)+ny]) * 0.1;
  }

  // middle boxes
  #pragma vector aligned
  for (int i = 2; i < nx+1; ++i) {
    for (int j = 1; j < ny-1; ++j) {
      tmp_image[j+i*ny] = image[j+i*ny] * 0.6 + (image[j+i*ny-ny] + image[j+i*ny+ny] + image[j+i*ny-1] + image[j+i*ny+1]) * 0.1;
    }
  }
}

void stencil_last(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  // Case last column first row
  tmp_image[ny*(nx)] = image[ny*(nx)] * 0.6 + (image[ny*(nx)+1] + image[ny*(nx)-ny]) * 0.1;

  // Case last column last row
  tmp_image[((nx+1)*ny)-1] = image[((nx+1)*ny)-1] * 0.6 + (image[((nx+1)*ny)-2] + image[((nx+1)*ny)-ny]) * 0.1;

  // last column
  for (int i = 1; i < ny-2; ++i) {
    tmp_image[i+((nx-2)*ny)] = image[i+((nx-2)*ny)] * 0.6 + (image[i+((nx-2)*ny)-1] + image[i+((nx-2)*ny)+1] + image[i+((nx-2)*ny)-ny]) * 0.1;
  }

  // first row
  for (int i = 1; i < nx+1; ++i) {
    tmp_image[i*ny] = image[i*ny] * 0.6 + (image[(i*ny)+1] + image[(i*ny)+ny] + image[(i*ny)-ny]) * 0.1;
  }

  // last row
  for (int i = 1; i < nx+1; ++i) {
    tmp_image[i*ny+(ny-1)] = image[i*ny+(ny-1)] * 0.6 + (image[i*ny+(ny-1)-1] + image[i*ny+(ny-1)-ny] + image[i*ny+(ny-1)+ny]) * 0.1;
  }

  // middle boxes
  #pragma vector aligned
  for (int i = 1; i < nx+1; ++i) {
    for (int j = 1; j < ny-1; ++j) {
      tmp_image[j+i*ny] = image[j+i*ny] * 0.6 + (image[j+i*ny-ny] + image[j+i*ny+ny] + image[j+i*ny-1] + image[j+i*ny+1]) * 0.1;
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, float *  image, float *  tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}

int calc_ncols_from_rank(int rank, int size, int ny)
{
  int ncols;

  ncols = ny / size;       /* integer division */
  if ((ny % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += ny % size;  /* add remainder to last rank */
  }
  
  return ncols;
}
# GPU-vs-GPU-Connect-4-Game-Using-CUDA-Parallel-ComputingProject Overview

# PROJECT OVERVIEW:

This project demonstrates a GPU-accelerated Connect 4 game where two GPU-based competitors play against each other using CUDA parallel computing. The system uses CUDA kernels to evaluate possible moves in parallel, allowing faster and more efficient game decision-making compared to traditional sequential execution.

The project showcases how GPUs can be used not only for graphics rendering but also for artificial intelligence and game strategy evaluation. Each GPU player uses heuristic-based decision logic to determine the best move during its turn.

The CPU acts as the game controller, while CUDA-enabled GPU kernels perform move analysis and scoring in parallel threads.

This project highlights important concepts such as:

CUDA kernel execution
Parallel processing
GPU memory management
Thread-based computation
Heuristic game AI

# Technologies Used
Technology	Purpose
CUDA	Parallel GPU programming
C++	Core game implementation
NVIDIA GPU	Hardware acceleration
Google Colab	Cloud-based CUDA execution
NVCC Compiler	CUDA code compilation
CPU-GPU Memory Transfer	Data communication

# Project Structure:

```
Project Folder
│
├── connect4.cu
│     Main CUDA source code
│
├── CUDA Kernels
│     Parallel move evaluation
│
├── Game Logic
│     Board management
│     Winner checking
│     Turn handling
│
├── GPU Memory Management
│     cudaMalloc()
│     cudaMemcpy()
│     cudaFree()
│
└── Output Console
      Game board visualization
      Winner announcement
```

# PROGRAM:
```
%%writefile connect4.cu

#include <iostream>
#include <cuda.h>

using namespace std;

#define ROWS 6
#define COLS 7

// ----------------------------
// Print Board
// ----------------------------
void printBoard(int board[ROWS][COLS])
{
    cout << "\n";

    for(int i=0;i<ROWS;i++)
    {
        for(int j=0;j<COLS;j++)
        {
            cout << board[i][j] << " ";
        }
        cout << endl;
    }

    cout << "--------------\n";
}

// ----------------------------
// Check Valid Move
// ----------------------------
bool validMove(int board[ROWS][COLS], int col)
{
    return board[0][col] == 0;
}

// ----------------------------
// Drop Piece
// ----------------------------
void dropPiece(int board[ROWS][COLS], int col, int player)
{
    for(int i=ROWS-1;i>=0;i--)
    {
        if(board[i][col] == 0)
        {
            board[i][col] = player;
            break;
        }
    }
}

// ----------------------------
// Winner Check
// ----------------------------
bool checkWinner(int board[ROWS][COLS], int player)
{
    // Horizontal
    for(int r=0;r<ROWS;r++)
    {
        for(int c=0;c<COLS-3;c++)
        {
            if(board[r][c]==player &&
               board[r][c+1]==player &&
               board[r][c+2]==player &&
               board[r][c+3]==player)
                return true;
        }
    }

    // Vertical
    for(int c=0;c<COLS;c++)
    {
        for(int r=0;r<ROWS-3;r++)
        {
            if(board[r][c]==player &&
               board[r+1][c]==player &&
               board[r+2][c]==player &&
               board[r+3][c]==player)
                return true;
        }
    }

    return false;
}

// ----------------------------
// GPU Move Evaluation Kernel
// ----------------------------
__global__ void evaluateMoves(int *board, int *scores, int player)
{
    int col = threadIdx.x;

    if(col >= COLS)
        return;

    // Simple heuristic:
    // prefer center columns

    int center = 3;

    scores[col] = 10 - abs(center - col);

    // Penalize full columns
    if(board[col] != 0)
        scores[col] = -1000;
}

// ----------------------------
// Get Best Move From GPU
// ----------------------------
int gpuMove(int board[ROWS][COLS], int player)
{
    int h_board[ROWS][COLS];

    for(int i=0;i<ROWS;i++)
        for(int j=0;j<COLS;j++)
            h_board[i][j] = board[i][j];

    int *d_board;
    int *d_scores;

    cudaMalloc((void**)&d_board, sizeof(int)*ROWS*COLS);
    cudaMalloc((void**)&d_scores, sizeof(int)*COLS);

    cudaMemcpy(
        d_board,
        h_board,
        sizeof(int)*ROWS*COLS,
        cudaMemcpyHostToDevice
    );

    evaluateMoves<<<1, COLS>>>(d_board, d_scores, player);

    int h_scores[COLS];

    cudaMemcpy(
        h_scores,
        d_scores,
        sizeof(int)*COLS,
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_board);
    cudaFree(d_scores);

    int bestCol = 0;
    int bestScore = -9999;

    for(int i=0;i<COLS;i++)
    {
        if(h_scores[i] > bestScore &&
           validMove(board, i))
        {
            bestScore = h_scores[i];
            bestCol = i;
        }
    }

    return bestCol;
}

// ----------------------------
// Main
// ----------------------------
int main()
{
    int board[ROWS][COLS] = {0};

    int currentPlayer = 1;

    int moves = 0;

    while(moves < 42)
    {
        printBoard(board);

        int move = gpuMove(board, currentPlayer);

        cout << "GPU Player "
             << currentPlayer
             << " chooses column "
             << move
             << endl;

        dropPiece(board, move, currentPlayer);

        if(checkWinner(board, currentPlayer))
        {
            printBoard(board);

            cout << "GPU Player "
                 << currentPlayer
                 << " WINS!\n";

            break;
        }

        currentPlayer =
            (currentPlayer == 1) ? 2 : 1;

        moves++;
    }

    return 0;
}
```

# OUTPUT:

<img width="421" height="804" alt="image" src="https://github.com/user-attachments/assets/2800a0af-2a03-498d-9c7e-2f21f72fc674" />

# RESULT:

This project proves that CUDA-enabled GPUs can effectively handle game AI and decision-making tasks using parallel computing techniques. The implementation demonstrates how GPUs can accelerate move evaluation and support competitive AI gameplay in strategy-based games like Connect 4

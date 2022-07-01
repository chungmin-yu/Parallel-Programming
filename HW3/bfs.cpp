#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *TD_frontier,
    vertex_set *TD_new_frontier,
    int *distances,
    bool *change, bool *BU_frontier)
{
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < TD_frontier->count; i++)
    {
        int node = TD_frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            //distances[outgoing] = distances[node] + 1;
            //int index = new_frontier->count++;
            //new_frontier->vertices[index] = outgoing;

            if(__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node]+1))
            {
                int index = __sync_fetch_and_add(&TD_new_frontier->count, 1);
                TD_new_frontier->vertices[index] = outgoing;
                BU_frontier[outgoing] = 1;
                *change = true;
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *TD_frontier = &list1;
    vertex_set *TD_new_frontier = &list2;

    bool *BU_frontier = (bool*)calloc(graph->num_nodes , sizeof(bool));
    bool change = false;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    TD_frontier->vertices[TD_frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (TD_frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(TD_new_frontier);

        top_down_step(graph, TD_frontier, TD_new_frontier, sol->distances, &change, BU_frontier);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", TD_frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = TD_frontier;
        TD_frontier = TD_new_frontier;
        TD_new_frontier = tmp;
    }
    free(BU_frontier);
}

void bottom_up_step(
    Graph g,
    bool *BU_frontier,
    bool *BU_new_frontier,
    int *distances,
    bool *change, int dis)
{
    //bool change = false;

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0 ; i < g->num_nodes ; i++)
    {
        if(distances[i] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                                ? g->num_edges 
                                : g->incoming_starts[i + 1];
            
            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge ; neighbor < end_edge ; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];

                if (BU_frontier[incoming])
                {
                    BU_new_frontier[i] = 1;
                    distances[i] = dis;
                    *change = true;
                    break;
                }
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    bool *BU_frontier = (bool*)calloc(graph->num_nodes , sizeof(bool));
    bool *BU_new_frontier = (bool*)calloc(graph->num_nodes , sizeof(bool));

    int dis = 1;
    bool change = true;    

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0 ; i < graph->num_nodes ; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    BU_frontier[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;    
    


    while(change)
    {


#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        change=false;
        bottom_up_step(graph, BU_frontier, BU_new_frontier, sol->distances, &change, dis);
        
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
#endif

        // swap pointers
        bool *tmp = BU_frontier;
        BU_frontier = BU_new_frontier;
        BU_new_frontier = tmp;
        dis++;
    }
    free(BU_frontier);
    free(BU_new_frontier);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *TD_frontier = &list1;
    vertex_set *TD_new_frontier = &list2;

    bool *BU_frontier = (bool*)calloc(graph->num_nodes, sizeof(bool));
    bool *BU_new_frontier = (bool*)calloc(graph->num_nodes, sizeof(bool));

    int dis = 1;
    bool change = true;    

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    TD_frontier->vertices[TD_frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    bool BU = false;

    while(change)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        change = false;
        if((float)(TD_frontier->count)/(float)(graph->num_nodes) < 0.1){
            top_down_step(graph, TD_frontier, TD_new_frontier, sol->distances, &change, BU_frontier);
            BU=false;
        }
        else{
            bottom_up_step(graph, BU_frontier, BU_new_frontier, sol->distances, &change, dis);
            BU = true;
        }


#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
#endif        

        // swap pointers
        if(!BU){
            vertex_set *tmp = TD_frontier;
            TD_frontier = TD_new_frontier;
            TD_new_frontier = tmp;

        }else{
            bool *tmp = BU_frontier;
            BU_frontier = BU_new_frontier;
            BU_new_frontier = tmp;
        }

        dis++;
    }
    free(BU_frontier);
    free(BU_new_frontier);    
}


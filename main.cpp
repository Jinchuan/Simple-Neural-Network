//
//  main.cpp
//  Flower_Classification_Backpropagation_2
//
//  Created by Jinchuan Shi on 11/22/12.
//  Copyright (c) 2012 Jinchuan Shi. All rights reserved.
//
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "stdbool.h"
#include <fstream>

using namespace std;

double** iris_data_parser(const char* filename)
{
    char temp_data[200];
    
    int line_number;
    
    double *seple_length, *sepal_width, *petal_length, *petal_width, *iris_classification;
    char classification[30];
    
    double **iris_matrix;
    FILE *fp;
	
    if(!(fp = fopen(filename,"r")))
    {
        printf("\n Error : Unable to open file");
        exit(1);
    }
    
    while(!feof(fp))
    {
        fscanf(fp,"%s",&temp_data);
        
        if(strcmp(temp_data,"Line_Number:")==0)
        {
            fscanf(fp,"%d", &line_number);
            
            //printf("line number is %d", line_number);
            
            seple_length = (double*) malloc(sizeof(double) * line_number);
            sepal_width = (double*) malloc(sizeof(double) * line_number);
            petal_length = (double*) malloc(sizeof(double) * line_number);
            petal_width = (double*) malloc(sizeof(double) * line_number);
            iris_classification = (double*) malloc ( sizeof(double) * line_number);
            
            iris_matrix = (double**) malloc(sizeof(double*) * line_number);
            
            for(int i = 0; i < line_number; i++)
            {
                
                fscanf(fp,"%lf,%lf,%lf,%lf,%s", &seple_length[i],&sepal_width[i],&petal_length[i],&petal_width[i],&classification);
                
                if (strcmp(classification,"Iris-setosa")==0 )
                {
                    iris_classification[i] = 0;
                }
                else if (strcmp(classification,"Iris-versicolor")==0)
                {
                    iris_classification[i] = 0.5;
                }
                else if (strcmp(classification,"Iris-virginica")==0)
                {
                    iris_classification[i] = 1;
                }
            }
        }
        else if(strcmp(temp_data,"EOF")==0)
            break;
        
    }
    
    for(int i = 0;i<line_number;i++)
    {
        iris_matrix[i] = (double*) malloc(sizeof(double) * 5);
        
        iris_matrix[i][0] = seple_length[i];
        iris_matrix[i][1] = sepal_width[i];
        iris_matrix[i][2] = petal_length[i];
        iris_matrix[i][3] = petal_width[i];
        iris_matrix[i][4] = iris_classification[i];
    }
    return(iris_matrix);
}

int generateRandomNumber(int range)
{
    int random_Number = rand() % range;
    return random_Number;
}

double** initial_weight(int number1, int number2)
{
    double** weight;
    weight = (double**)malloc(sizeof(double*) * number1);
    
    for (int i = 0; i < number1; i++)
    {
        weight[i] = (double*) malloc((sizeof(double) * number2));
    }
    
    for(int i = 0; i<number1; i++)
    {
        for( int j = 0; j<number2; j++)
        {
            int positive_nagetive = generateRandomNumber(2);
            if (positive_nagetive == 1)
            {
                weight[i][j] = generateRandomNumber(12)/10.0;
            }
            else
            {
                weight[i][j] = (generateRandomNumber(12)/10.0) * (-1);
            }
        }
    }
    return weight;
}

double** initial_delta_weight(int number1, int number2)
{
    double** weight;
    weight = (double**)malloc(sizeof(double*) * number1);
    
    for (int i = 0; i < number1; i++)
    {
        weight[i] = (double*) malloc((sizeof(double) * number2));
    }
    for(int i = 0; i<number1; i++)
    {
        for( int j = 0; j<number2; j++)
        {
            
            weight[i][j] = 0.0;
        }
    }
    return weight;
}


double* initial_thet(int number)
{
    double* thet;
    thet = (double*)malloc(sizeof(double*) * number);
    
    for(int i = 0; i<number; i++)
    {
        int positive_nagetive = generateRandomNumber(2);
        if (positive_nagetive == 1)
        {
            thet[i] = generateRandomNumber(12)/10.0;
        }
        else
        {
            thet[i] = (generateRandomNumber(12)/10.0) * (-1);
        }
    }
    return thet;
}

double* initial_delta_thet(int number)
{
    double* thet;
    thet = (double*)malloc(sizeof(double*) * number);
    
    for(int i = 0; i<number; i++)
    {
        
        thet[i] = 0.0;
    }
    return thet;
}


double calculator_hidden_output(double **input_matrax, double **weight_matrix, int outside_loop ,int output_index,int input_number, double *thet)
{
    double result = 0.0;
    for (int i = 0; i < input_number; i++)
    {
        result = result + input_matrax[outside_loop][i] * weight_matrix[i][output_index];
    }
    result = result - thet[output_index];
    
    result = 1 / (1 + exp(-(result)));
    
    return result;
    
}

double calculator_output(double *input_matrax, double **weight_matrix,int output_index,int input_number, double *thet)
{
    double result = 0.0;
    for (int i = 0; i < input_number; i++)
    {
        result = result + input_matrax[i] * weight_matrix[i][output_index];
    }
    result = result - thet[output_index];
    
    result = 1 / (1 + exp(-(result)));
    
    return result;
    
}




int main()
{
    //read the input data
    double **iris_matrix;
    iris_matrix = iris_data_parser("iris.txt");
    
    //split example data to train, tuning and testing
    double **iris_training_set;
    iris_training_set = (double**)malloc(sizeof(double*) * 75);
    for (int i = 0 ; i < 75; i ++){
        iris_training_set[i] = (double*)malloc(sizeof(double) * 5);
    }
    
    double **iris_tuning_set = (double**)malloc(sizeof(double*) * 37);
    for (int i = 0; i < 37; i++){
        iris_tuning_set[i] = (double*) malloc(sizeof(double) * 5);
    }
    
    double **iris_test_set = (double**) malloc(sizeof(double*) * 38);
    for (int i = 0 ; i < 38 ; i++){
        iris_test_set[i] = (double*) malloc(sizeof(double) * 5);
    }
        
    
    int train_number = 0;
    int tuning_number = 0;
    int test_number = 0;
    for(int i = 0; i < 148; i = i + 4)
    {
        for(int j = 0; j < 5; j++)
        {
            iris_training_set[train_number][j] = iris_matrix[i][j];
            iris_training_set[train_number + 1][j] = iris_matrix[i+1][j];
            iris_tuning_set[tuning_number][j] = iris_matrix[i+2][j];
            iris_test_set[test_number][j] = iris_matrix[i+3][j];
        }
        train_number += 2;
        tuning_number++;
        test_number++;
    }
    for(int i = 0; i< 5; i++)
    {
        iris_training_set[74][i] = iris_matrix[148][i];
        iris_test_set[37][i] = iris_matrix[149][i];
    }
    
    
    
    printf("training set: \n");
    for (int i = 0; i < 75; i++)
    {
        for(int j = 0; j <5; j++)
        {
            printf("%.1f,", iris_training_set[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    //
    //    printf("tuning set: \n");
    //    for (int i = 0; i < 37; i++)
    //    {
    //        for(int j = 0; j <5; j++)
    //        {
    //            printf("%.1f,", iris_tuning_set[i][j]);
    //        }
    //        printf("\n");
    //    }
    //
        printf("test set: \n");
        for (int i = 0; i < 38; i++)
        {
            for(int j = 0; j <5; j++)
            {
                printf("%.1f,", iris_test_set[i][j]);
            }
            printf("\n");
        }
    printf("\n");
    
    // learning rate
    double alpha = 0.1;
    
    // network set up
    int input_node_number = 4;
    int hidden_node_number = 3;
    int output_node_number = 1;
    
    // initial random weight at beginning
    srand((unsigned)time(0));
    double** input_weight_matrax = initial_weight(input_node_number, hidden_node_number);
    for (int j = 0; j < hidden_node_number; j++){
        for( int i = 0; i< input_node_number; i++){
            printf("input weight %d,%d = %f  ", i, j, input_weight_matrax[i][j]);
        }
        printf(" \n  ");
    }
    
    double **hidden_weight_matrax = initial_weight(hidden_node_number, output_node_number);    
    for (int j = 0; j < output_node_number; j++){
        for( int i = 0; i< hidden_node_number; i++){
            printf("hidden weight %d,%d = %f  ", i, j, hidden_weight_matrax[i][j]);
        }
        printf(" \n  ");
    }
    
    
    double *hidden_thet_array = initial_thet(hidden_node_number);
    double *output_thet_array = initial_thet(output_node_number);
        
    // calculate the output
    double mean_square_error = 0.01;
    int a = 0;
    while (mean_square_error > 0.005)
    {
        a++;
        mean_square_error = 0.0;
        for(int outside_loop = 0; outside_loop < 75; outside_loop++)
        {
            double *hidden_output = (double*) malloc(sizeof(double) * hidden_node_number);
            for ( int i = 0 ; i < hidden_node_number ; i++){
                hidden_output[i] = 0.0;
            }
            
            double *output = (double*) malloc(sizeof(double) * output_node_number);
            for ( int i = 0 ; i < output_node_number ; i++){
                output[i] = 0.0;
            }
            
            // calculator the hidden output
            for ( int i = 0 ; i < hidden_node_number ; i++){
                hidden_output[i] = calculator_hidden_output(iris_training_set, input_weight_matrax, outside_loop, i, input_node_number, hidden_thet_array);
            }
            
            // calculator the output
            for (int i = 0 ; i < output_node_number ; i++){
                output[i] = calculator_output(hidden_output, hidden_weight_matrax, i , hidden_node_number, output_thet_array);
            }
            
            // the error of all output level
            double *error = (double*) malloc(sizeof(double) * output_node_number);
            for(int i = 0; i < output_node_number; i ++){
                error[i] = iris_training_set[outside_loop][4]/output_node_number - output[i];
            }
            
            double square_error = 0.0;
            for (int i = 0; i < output_node_number; i++){
                square_error = square_error + pow(error[i],2);
            }
            
            mean_square_error = mean_square_error + square_error;
            
            // calculator the output level delta
            double *delta_output_level = (double*) malloc(sizeof(double) * output_node_number);
            for (int i = 0 ; i < output_node_number; i++){
                delta_output_level[i] = output[i] * (1 - output[i]) * error[i];
            }
            
            //the weight corrections, alpha = 0.1
            double **hidden_delta_weight_matrax = initial_delta_weight(hidden_node_number, output_node_number);
            for(int i = 0; i < hidden_node_number; i++){
                for (int j = 0; j < output_node_number; j++){
                    hidden_delta_weight_matrax[i][j] = alpha * hidden_output[i] * delta_output_level[j];
                }
            }
            double *delta_output_thet_array = initial_delta_thet(output_node_number);
            for (int i = 0; i < output_node_number; i++){
                delta_output_thet_array[i] = alpha *(-1) * delta_output_level[i];
            }
            
            // for each hidden unit h
            double *delta_hidden_level = (double*) malloc( sizeof(double) * hidden_node_number);
            for ( int i = 0; i < hidden_node_number ; i++){
                for (int j = 0 ; j < output_node_number; j++){
                    delta_hidden_level[i] = delta_hidden_level[i] + hidden_output[i] * (1 - hidden_output[i]) * delta_output_level[j] * hidden_weight_matrax[i][j];
                }
            }
            
            // the input weight corrections, alpha = 0.1            
            double **input_delta_weight_matrax = initial_delta_weight(input_node_number, hidden_node_number);
            for(int i = 0; i < input_node_number; i++){
                for (int j = 0; j < hidden_node_number; j++){
                    input_delta_weight_matrax[i][j] = alpha * iris_training_set[outside_loop][i] * delta_hidden_level[j];
                }
            }
            double *delta_hidden_thet_array = initial_delta_thet(hidden_node_number);
            for (int i = 0; i < hidden_node_number; i++){
                delta_hidden_thet_array[i] = alpha *(-1) * delta_hidden_level[i];
            }
            
            // updata all weights and thresholds
            for (int j = 0; j < hidden_node_number; j++){
                for( int i = 0; i< input_node_number; i++){
                    input_weight_matrax[i][j] = input_weight_matrax[i][j] + input_delta_weight_matrax[i][j];
                }
            } 
            for(int j = 0; j < output_node_number; j++){
                for(int i = 0; i < hidden_node_number; i++){
                    hidden_weight_matrax[i][j] = hidden_weight_matrax[i][j] + hidden_delta_weight_matrax[i][j];
                }
            }
            for (int i = 0; i < output_node_number; i++){
                output_thet_array[i] = output_thet_array[i] + delta_output_thet_array[i];
            }
            for (int i = 0; i < hidden_node_number; i++){
                hidden_thet_array[i] = hidden_thet_array[i] + delta_hidden_thet_array[i];
            }
        }
        mean_square_error = mean_square_error / 75;
        printf("mean square error %d = %f \n", a, mean_square_error);
        
        
        // tuning data mean square error
        double mean_square_error_tunning = 0.0;
        for (int tuning_loop = 0; tuning_loop < 37; tuning_loop++)
        {
            double *hidden_output = (double*) malloc(sizeof(double) * hidden_node_number);
            for ( int i = 0 ; i < hidden_node_number ; i++)
            {
                hidden_output[i] = 0.0;
            }
            
            double *output = (double*) malloc(sizeof(double) * output_node_number);
            for ( int i = 0 ; i < output_node_number ; i++)
            {
                output[i] = 0.0;
            }
            
            
            // calculator the hidden output
            for ( int i = 0 ; i < hidden_node_number ; i++)
            {
                hidden_output[i] = calculator_hidden_output(iris_tuning_set, input_weight_matrax, tuning_loop, i, input_node_number, hidden_thet_array);
            }
            
            // calculator the output
            for (int i = 0 ; i < output_node_number ; i++)
            {
                output[i] = calculator_output(hidden_output, hidden_weight_matrax, i , hidden_node_number, output_thet_array);
            }
            
            double *error = (double*) malloc(sizeof(double) * output_node_number);
            for(int i = 0; i < output_node_number; i ++)
            {
                error[i] = iris_tuning_set[tuning_loop][4]/output_node_number - output[i];
            }
            
            double square_error = 0.0;
            for (int i = 0; i < output_node_number; i++)
            {
                square_error = square_error + pow(error[i],2);
            }
            
            mean_square_error_tunning = mean_square_error_tunning + square_error;

        }
        mean_square_error_tunning = mean_square_error_tunning /37;
        printf("  mean_square_error_tunning = %f", mean_square_error_tunning);

        
        
    }
    
    
    // display the weights and thets after training
    for (int j = 0; j < hidden_node_number; j++)
    {
        for( int i = 0; i< input_node_number; i++)
        {
            printf("input weight %d,%d = %f  ", i, j, input_weight_matrax[i][j]);
        }
        printf(" \n  ");
    }
        printf(" \n  ");
    for (int j = 0; j < output_node_number; j++)
    {
        for( int i = 0; i< hidden_node_number; i++)
        {
            printf("hidden weight %d,%d = %f  ", i, j, hidden_weight_matrax[i][j]);
        }
        printf(" \n  ");
    }
    printf(" \n  ");
    for (int i = 0; i< hidden_node_number; i++)
    {
        printf("hidden node thet %d: %f\n",i, hidden_thet_array[i]);
    }
    for (int i = 0; i< output_node_number; i++)
    {
        printf("output node thet %d: %f\n",i, output_thet_array[i]);
    }
    
    printf("test result: \n\n");
    
    // display the train set output after traing
    for (int training_loop = 0; training_loop < 75; training_loop++)
    {
        double *hidden_output = (double*) malloc(sizeof(double) * hidden_node_number);
        for ( int i = 0 ; i < hidden_node_number ; i++)
        {
            hidden_output[i] = 0.0;
        }
        double *output = (double*) malloc(sizeof(double) * output_node_number);
        for ( int i = 0 ; i < output_node_number ; i++)
        {
            output[i] = 0.0;
        }
        // calculator the hidden output
        for ( int i = 0 ; i < hidden_node_number ; i++)
        {
            hidden_output[i] = calculator_hidden_output(iris_training_set, input_weight_matrax, training_loop, i, input_node_number, hidden_thet_array);
        }
        // calculator the output
        for (int i = 0 ; i < output_node_number ; i++)
        {
            output[i] = calculator_output(hidden_output, hidden_weight_matrax, i , hidden_node_number, output_thet_array);
        }
        for (int i = 0 ; i < output_node_number; i++)
        {
            printf("Training output[%d] = %f \n", i, output[i]);
        }
    }
    
    
    // display the test set output after training 
    for (int outside_loop_test = 0; outside_loop_test < 38; outside_loop_test++)
    {
        double *hidden_output = (double*) malloc(sizeof(double) * hidden_node_number);
        for ( int i = 0 ; i < hidden_node_number ; i++)
        {
            hidden_output[i] = calculator_hidden_output(iris_test_set, input_weight_matrax, outside_loop_test, i, input_node_number, hidden_thet_array);
        }
        double *output = (double*) malloc(sizeof(double) * output_node_number);
        for (int i = 0 ; i < output_node_number ; i++)
        {
            output[i] = calculator_output(hidden_output, hidden_weight_matrax, i , hidden_node_number, output_thet_array);
        }
        for (int i = 0 ; i < output_node_number; i++)
        {
            printf(" Test output[%d] = %f \n", i, output[i]);
        }
    }
    
    return 0;
}










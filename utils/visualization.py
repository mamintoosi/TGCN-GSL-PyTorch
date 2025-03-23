#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_result(y_pred, y_true, plot_file):
    """
    Plot the prediction results.
    
    Args:
        y_pred: numpy array of shape (batch_size * pre_len, num_nodes)
        y_true: numpy array of shape (batch_size * pre_len, num_nodes)
        metrics_file: path to the metrics file, used to generate the plot filename
    """
    # Create figure
    # fig = plt.figure(figsize=(15, 10))
    
    # # Plot all test data
    # plt.subplot(2, 1, 1)
    # plt.plot(y_true[:, 0], label='True', color='blue', alpha=0.7)
    # plt.plot(y_pred[:, 0], label='Predicted', color='red', alpha=0.7)
    # plt.title('All Test Data')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Traffic Flow')
    # plt.legend()
    # plt.grid(True)
    
    # Plot one day's worth of test data
    # plt.subplot(2, 1, 2)

    num_test_data_points = 192

    # Save predictions and true values to Excel file
    result_file = plot_file.replace('.pdf', '.xlsx')
    df_result = pd.DataFrame({
        'True': y_true[:num_test_data_points, 0],
        'Predicted': y_pred[:num_test_data_points, 0]
    })
    
    with pd.ExcelWriter(result_file) as writer:
        df_result.to_excel(writer, sheet_name='Results', index=False)

    plt.plot(y_true[:num_test_data_points, 0], label='True', color='blue', alpha=0.7)
    plt.plot(y_pred[:num_test_data_points, 0], label='Predicted', color='red', alpha=0.7)
    plt.title('Test Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic Flow')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(plot_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path):
    ###train_rmse & test_rmse 
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.jpg')
    plt.show()
    #### train_loss & train_rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.jpg')
    plt.show()

    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse,'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse.jpg')
    plt.show()

    ### accuracy
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.jpg')
    plt.show()
    ### rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse.jpg')
    plt.show()
    ### mae
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.jpg')
    plt.show()



%**************************************************************************
% 
% Master OSS
% 
% Pattern Recognition
% 
%**************************************************************************

clc;
clear;
close all;

N = 1000 ;
Vec = rand(1,N);

VecDelta = Vec < 0.01 ;


step = 0.05;
 Kernel = 0.5*[0:step:1 (1-step):-step:0 ] .^2 ;

figure;
stem(Kernel,'.')


figure;
subplot(211);stem(Vec,'.')
subplot(212);stem(VecDelta,'.')



PDF = filter(Kernel , 1 , VecDelta) ;

figure;
stem(PDF,'.')












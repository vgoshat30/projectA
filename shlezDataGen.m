%% Training and testiong data generation-asymptotic massive MIMO case study
% Created April 2018
% 
% Creator: Nir Shlezinger
%
% Modified 10 May 2018
% Gosha Tsintsadze
% Matan Shohat
%
%   Update description:
%       Changing the code to use only for generating .mat file with
%       training and testing X and S datasets (to be used later in python
%       based NN).

clear all;
close all;
clc;
%% Parameters setting
s_fPower = 4;
s_fNu = 4;
s_fNt = 10;
s_fRatio = 3;

s_fT = 2^15; % number of training samples
s_fD = 2^10; % number of data samples
%% Generate training data and pilot matrix
% Pilots matrix
s_fTau = (s_fNu*s_fRatio);  
m_fPhi = dftmtx(s_fTau);
m_fPhi = m_fPhi(:,1:s_fNu);
m_fSigmaT = eye(s_fTau) + s_fPower*(m_fPhi*m_fPhi');
m_fLMMSE =  (sqrt(s_fPower) / (1 + s_fPower*s_fTau))*(kron(m_fPhi',eye(s_fNt)));

% Training  and data - generate channels and observations
m_fH = (1 / sqrt(2)) * (randn(s_fNu * s_fNt, s_fT + s_fD) + 1j*randn(s_fNu * s_fNt, s_fT + s_fD));
m_fW = (1 / sqrt(2)) * (randn(s_fTau * s_fNt, s_fT + s_fD) + 1j*randn(s_fTau * s_fNt, s_fT + s_fD));
m_fY = sqrt(s_fPower) *(kron(m_fPhi, eye(s_fNt))) * m_fH + m_fW;
% MMSE estimate
m_fHtilde = m_fLMMSE * m_fY;

% Convert to real valued training
trainS = [real(m_fH(:,1:s_fT)); imag(m_fH(:,1:s_fT))].';
trainX = [real(m_fY(:,1:s_fT)); imag(m_fY(:,1:s_fT))].';
% Convert to real valued data
dataS = [real(m_fH(:,s_fT+1:end)); imag(m_fH(:,s_fT+1:end))].';
dataX = [real(m_fY(:,s_fT+1:end)); imag(m_fY(:,s_fT+1:end))].';
%% Saving the mat file to manually chosen folder

shlezFolder = uigetdir(pwd,'Data Output Folder');
shlezMatFile = fullfile(shlezFolder,'shlezingerMat.mat');
% Check and nofify if file exsists
if exist(shlezMatFile,'file')
    answer = questdlg(['File named "' shlezMatFile ...
                       '.eps" already exists.' ...
                       ' Do you want to replace it?'], ...
                      'File Exists', ...
                      'Replace','Cancel','Replace');
    switch answer
        case 'Replace'
            save(shlezMatFile,'dataX','dataS','trainX','trainS');
        case 'Cancel'
            return;
    end
else
    save(shlezMatFile,'dataX','dataS','trainX','trainS');
end


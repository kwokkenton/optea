% OPT Forward Model MATLAB Implementation
% 
% 

close all
% Acquisition Parameters
N_pixels = 1040; % horizontal number of pixels 
e = 6.45e-3; % pixel size (mm)
N_angles = 400; % number of angles in tomographic reconstruction

% Optical parameters
w0 = 6.92e-6; % Minimum beam waist in Gaussian model (m)
lamb = 525e-9; % (m)

% Define Reconstruction object
object_name = 'beads';
value = 10;

switch object_name
    % Case 1: Shepp-logan phantom
    case 'shepp'
        shepp_logan = phantom(N_pixels); 
    case 'beads'
        % Case 2: Simulated bead phantom====
        object = zeros(N_pixels, N_pixels); 
        for idx= 520:40:1000 % generate point objects at these equally spaced locations
            object(520, idx) = value;
            object(520, idx+1) = value;
            object(521, idx) = value;
            object(521, idx+1) = value;
        end
    case 'image'
        watermelon = imresize(imread("map.jpg"), 1/13);
        watermelon = padarray(255-rgb2gray(watermelon), [513, 513]);
        object = (cast(watermelon(1:1040 ,1:1040), 'double')/255) ;
    
end

imshow(object)
%% Generates/ loads in MTF filter

focal_plane_shift = 0;  %(mm)
filter_name = 'f6'; 

% parameters
defocuses = linspace(-N_pixels*e/2, N_pixels*e/2, N_pixels); % defocuses (mm)
nyquist_freq = 1/(2*e);
frequencies = (-N_pixels/2:N_pixels/2-1)/(e*N_pixels);

% Read in MTF from .mat
S = load('mtf_filters.mat');

switch filter_name
    % These are experimental MTFs, tidied up in Python and imported %%%%%%%
    case 'f6'
        frequency_filter = S.f6;
    case 'f11'
        frequency_filter = S.f11;
    case 'f17'
        frequency_filter = S.f17;
    case 'f26'
        frequency_filter = S.f26;
    % These are simulated MTFs, generated in MATLAB %%%%%%%%%%%%%%%%%%%%%%%
    case 'gaussian'
        % creates Gaussian frequency filter
        frequency_filter = create_gaussian_filter(frequencies, defocuses, N_pixels, w0, lamb); 
    case 'shifted'
        % focal shifted mtf using Gaussian frequency filter
        frequency_filter = create_gaussian_filter(frequencies, defocuses-focal_plane_shift, N_pixels, w0, lamb); 
    case 'focal_scanning'
        % focal scanning Gaussian frequency filter (shifted and summed)
        shifts = 0:0.75:3;  % (mm)
        frequency_filter = create_gaussian_filter(frequencies, defocuses, N_pixels, w0, lamb)/length(shifts);
        for i= 0:length(shifts)-1
            focal_plane_shift = shifts(i+1);
            shifted_filter = create_gaussian_filter(frequencies, defocuses-focal_plane_shift, N_pixels, w0, lamb)/length(shifts);
            frequency_filter = frequency_filter + shifted_filter;
        end
    case 'no_filter'
        frequency_filter = ones(N_pixels, N_pixels);
end
%%
% Display frequency filter
h = pcolor(frequencies, defocuses,  frequency_filter);
% imshow(frequency_filter);
colormap(hot);
set(h, 'EdgeColor', 'none') % remove grid lines as they will make image black
xlabel('Spatial frequencies (lp/mm)')
ylabel('defocuses (mm)')
title(filter_name); % Place title here

% Save for future use
% save('standard_filter.mat','frequency_filter');

%% setup reconstruction variables
%filter = zeros(N_pixels, N_pixels); % 2D frequency filter, dimensions of (defocuses, spatial frequencies)
angles = 360/N_angles *(0:N_angles -1); % define acquisition angles
fftshifted_MTF = fftshift(frequency_filter,2); % fftshift in frequency direction

%% Run forward model
sino = forward(object, fftshifted_MTF, angles, N_pixels);

% Reconstruct OPT image
figure('Name', 'Reconstruction');
im = iradon(sino', -angles, 'linear','Ram-Lak',1,N_pixels);
imshow(im);
colormap(hot);
%%
imwrite(im,'f6_beads.tif');
%%
% N = 32;
% start_idx = N_pixels/2 - N/2+1;
% end_idx = start_idx + N -1; 
% x = im(start_idx:end_idx, start_idx:end_idx);
% I = B(start_idx:end_idx, start_idx:end_idx);
% save('object.mat','I', 'x');

%% Convert into Polars
% In case we wanted to see what the image looks like in polars
im_pad = padarray(im,[200 200]);
[a, b] = size(im);
imP = ImToPolar (im, 0, 1, b/2,2*a);
figure('Name', 'Polar Transform')
imshow(imP);
colormap(hot);
%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function sinogram = forward(slice, fftshifted_MTF, angles, N_pixels)
%     % slice is a 2D matrix
%     % fftshifted_MTF
%     % angles is an array of sampling angles
%     N_angles = length(angles);
%     sinogram = zeros(N_angles, N_pixels);
%     for idx = 1:N_angles
%         % rotates anticlockwise
%         rotated = imrotate(slice, angles(idx), 'crop');
%         % convolves each line in slice with PSF of that depth, in Fourier
%         % domain
%         fourier_image = fft(rotated, [], 2).* fftshifted_MTF ;
%         filtered = ifft(fourier_image, [], 2);
%         % then sums to get the projection (line integral)
%         projection = sum(filtered); 
%         sinogram(idx,:) = abs(projection);
%     end
% end

function frequency_filter = create_gaussian_filter(frequencies, defocuses, N_pixels, w0, lamb)
% CREATE_GAUSSIAN_FILTER: creates a Gaussian frequency filter
% frequencies are those sampled
% defocuses are in mm
% N_pixels 
% w0 is in m
% lamb is in m
    frequency_filter = zeros(N_pixels, N_pixels);
    %t = linspace(-N_pixels*e/2e3, N_pixels*2/2e3, N_pixels);
    beam_waist_depths = beam_waist(defocuses/1e3, w0, lamb); 
    sigma_prime = 1./(2* pi*beam_waist_depths)./1000; % units lp/mm
    for idx = 1:N_pixels
        frequency_filter(idx, :) = gaussian(frequencies, sigma_prime(idx));
    end
end

function profile = gaussian(x, sigma)
% obtains outputs of a Gaussian at values x, width sigma
% units of x and sigma must match
    profile = exp(-x.^2/(2*sigma^2));
end

function waist = beam_waist(s, w0, lamb)
    % computes beam waist at distance s from the focus
    % units of lamb, w0, s are in m
    waist = sqrt(w0^2 + (lamb*s/(pi*w0)).^2);
end
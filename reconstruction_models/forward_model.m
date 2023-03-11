% OPT Forward Model MATLAB Implementation
% 
% 
%
%
%

% Acquisition Parameters
N_pixels = 1040;
e = 6.45e-3; % pixel size (mm)
N_angles = 400;

% Aperture parameter
w0 = 6.92e-6; % Minimum beam waist (m)
lamb = 525e-9; % (m)

% Reconstruction object
P = phantom(N_pixels); % Shepp-logan phantom
B = zeros(N_pixels, N_pixels); % Simulated bead phantom
for idx= 520:50:1040
    B(520, idx) = 40;
end
imshow(B);
colormap(hot);

%% creates Gaussian frequency filter

defocuses = linspace(-N_pixels*e/2, N_pixels*e/2, N_pixels); % defocuses (mm)
nyquist_freq = 1/(2*e);
frequencies = linspace(-nyquist_freq, nyquist_freq, N_pixels);


% normal filter %%
%focal_plane_shift = 0;  %(mm)
%frequency_filter = create_gaussian_filter(frequencies, defocuses-focal_plane_shift, N_pixels, w0, lamb); 

% focal scanning Gaussian frequency filter %%%%%%
shifts = 0:0.75:3; 
frequency_filter = create_gaussian_filter(frequencies, defocuses, N_pixels, w0, lamb)/length(shifts);
for i= 0:length(shifts)-1
    focal_plane_shift = shifts(i+1);
    shifted_filter = create_gaussian_filter(frequencies, defocuses-focal_plane_shift, N_pixels, w0, lamb)/length(shifts);
    frequency_filter = frequency_filter + shifted_filter;
end

imshow(frequency_filter);
colormap(hot);

%% setup reconstruction variables
%filter = zeros(N_pixels, N_pixels); % 2D frequency filter, dimensions of (defocuses, spatial frequencies)
angles = 360/N_angles *(0:N_angles -1); % define acquisition angles
fftshifted_MTF = fftshift(frequency_filter,2); % fftshift in frequency direction


%% Run forward model
im = forward(B, fftshifted_MTF, angles, N_pixels);

% Reconstruct OPT image
imshow(iradon(im', -angles, 'linear','Ram-Lak',1,N_pixels));
colormap(hot)

%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sinogram = forward(slice, fftshifted_MTF, angles, N_pixels)
    % slice is a 2D matrix
    % fftshifted_MTF
    % angles is an array of sampling angles
    N_angles = length(angles);
    sinogram = zeros(N_angles, N_pixels);
    for idx = 1:N_angles
        % rotates anticlockwise
        rotated = imrotate(slice, angles(idx), 'crop');
        % convolves each line in slice with PSF of that depth, in Fourier
        % domain
        fourier_image = fft(rotated, [], 2).* fftshifted_MTF ;
        filtered = ifft(fourier_image, [], 2);
        % then sums to get the projection (line integral)
        projection = sum(filtered); 
        sinogram(idx,:) = abs(projection);
    end
end

function frequency_filter = create_gaussian_filter(frequencies, defocuses, N_pixels, w0, lamb)
    % creates a Gaussian frequency filter
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
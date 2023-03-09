% OPT Forward Model MATLAB Implementation

% Acquisition Parameters
N_pixels = 1040;
e = 6.45e-3; % pixel size (mm)
lamb = 525e-9; % (m)
N_angles = 400;

% Aperture parameter
w0 = 6.92e-6; % Minimum beam waist (m)

% Reconstruction object
P = phantom(N_pixels); 

nyquist_freq = 1/(2*e);
filter = zeros(N_pixels, N_pixels); % 2D frequency filter, dimensions of (defocuses, spatial frequencies)
defocuses = linspace(-N_pixels*e/2, N_pixels*e/2, N_pixels); % defocuses (mm)

frequencies = linspace(-nyquist_freq, nyquist_freq, N_pixels);

% creates Gaussian frequency filter
frequency_filter = create_gaussian_filter(frequencies, defocuses, N_pixels, w0, lamb); 
%imshow(frequency_filter);
%colormap(hot);

angles = 360/N_angles *(0:N_angles -1);
fftshifted_MTF = fftshift(frequency_filter,2); % fftshift in frequency direction
%
im = forward(P, fftshifted_MTF, angles, N_pixels);
imshow(iradon(im', -angles, 'linear','Ram-Lak',1,N_pixels));
colormap(hot);
% Functions
function sinogram = forward(slice, fftshifted_MTF, angles, N_pixels)
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
    profile = exp(-x.^2/(2*sigma^2));
end

function waist = beam_waist(s, w0, lamb)
    % computes beam waist at distance s from the focus
    waist = sqrt(w0^2 + (lamb*s/(pi*w0)).^2);
end
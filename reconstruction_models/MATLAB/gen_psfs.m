% Generate OPT PSFs
% Load in frequency filter
fftshifted_MTF = fftshift(load('standard_filter.mat').frequency_filter,2); % fftshift in frequency direction

% define acquisition angles
N_angles = 400; 
angles = 360/N_angles *(0:N_angles -1); 
N_pixels = 1040; % horizontal number of pixels 

%% Make impulse
impulse_value = 10; % arbitrary value

N = 64;
start_idx = N_pixels/2 - N/2+1;
end_idx = start_idx + N -1; 

% A = zeros(N*N);
A = (cast(zeros(N*N), 'single'));
tstart = tic;
for x_pos = start_idx:end_idx
    for y_pos = start_idx:end_idx
        tic
        % For each location
        % Create impulse response
        single_point = cast(zeros(N_pixels, N_pixels), 'single'); 
        single_point(y_pos, x_pos) = impulse_value;
        
        % Generate one reconstructed slice
        sino = forward(single_point, fftshifted_MTF, angles, N_pixels);
        im = iradon(sino', -angles, 'linear','Ram-Lak',1,N_pixels);
        im(im<0) = 0;
        % crop and extract PSF
        extracted = im(start_idx:end_idx, start_idx:end_idx); 
        image_col = x_pos-start_idx + 1;
        image_row = y_pos-start_idx + 1;
        index = N*(image_col - 1) + image_row; % correct index for wrapping due to reshaping
        A(:,index) = extracted(:);
        toc
        % print out to check progress
        index
        
    end
end
toc(tstart)
imshow(A);
%%
%save('simulation_output_64x64.mat','A');
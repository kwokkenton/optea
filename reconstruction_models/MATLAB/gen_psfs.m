% Generate OPT PSFs
% Load in frequency filter
% fftshifted_MTF = fftshift(load('standard_filter.mat').frequency_filter,2); % fftshift in frequency direction
fftshifted_MTF = fftshift(load('mtf_filters.mat').f26, 2) ;
% define acquisition angles
N_angles = 400; 
angles = 360/N_angles *(0:N_angles -1); 
N_pixels = 1040; % horizontal number of pixels 
%% Make impulse for generating the LR-PSFs
% impulse_value = 10; % arbitrary value
% 
% N = 64;
% start_idx = N_pixels/2 - N/2+1;
% end_idx = start_idx + N -1; 
% 
% % A = zeros(N*N);
% A = (cast(zeros(N*N), 'single'));
% tstart = tic;
% for x_pos = start_idx:end_idx
%     for y_pos = start_idx:end_idx
%         tic
%         % For each location
%         % Create impulse response
%         single_point = cast(zeros(N_pixels, N_pixels), 'single'); 
%         single_point(y_pos, x_pos) = impulse_value;
%         
%         % Generate one reconstructed slice
%         sino = forward(single_point, fftshifted_MTF, angles, N_pixels);
%         im = iradon(sino', -angles, 'linear','Ram-Lak',1,N_pixels);
%         im(im<0) = 0;
%         % crop and extract PSF
%         extracted = im(start_idx:end_idx, start_idx:end_idx); 
%         image_col = x_pos-start_idx + 1;
%         image_row = y_pos-start_idx + 1;
%         index = N*(image_col - 1) + image_row; % correct index for wrapping due to reshaping
%         A(:,index) = extracted(:);
%         toc
%         % print out to check progress
%         index
%         
%     end
% end
% toc(tstart)
% imshow(A);

%% Generate on-axis PSFs
y_pos = 520;
step = 40;
start_idx = 520;
end_idx = 1000;
num_beads = length(start_idx:step:end_idx);
impulse_value = 1e5; % arbitrary value
%%
% Generate PSFs on one radial arm
stack = (cast(zeros(num_beads, N_pixels,N_pixels), 'uint16'));
i = 1;
for x_pos = start_idx:step:end_idx
        tic
        i
        % For each location
        % Create impulse response
        single_point = cast(zeros(N_pixels, N_pixels), 'uint16'); 
%         single_point(y_pos, x_pos) = impulse_value;
        single_point(y_pos:y_pos+1, x_pos:x_pos+1) = impulse_value;
        
        % Generate one reconstructed slice
        sino = forward(single_point, fftshifted_MTF, angles, N_pixels);
        im = iradon(sino', -angles, 'linear','Ram-Lak',1, N_pixels);
        % Set unwanted values to 0
        im(im<0) = 0;
        % Add image to stack
        stack(i, :,:) = im;
        i = i+1; 
        toc
end
%%
% Display a sample image
imshow(squeeze(stack(2,:,:)))
%%
for i = 1:num_beads
    imwrite(squeeze(stack(i,:,:)),strcat('./simulated_f26/', num2str(i-1,'%04.f' ), '.tif'))
end
%%
%save('simulation_output_64x64.mat','A');
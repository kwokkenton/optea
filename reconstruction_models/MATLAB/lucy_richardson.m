% Implement Lucy Richardson algorithm for arbitrary blurring
% LR performs deconvolution with a simple update rule
% This should allow for a spatially varying PSF

%% LR Test cases
% Image dimensions
N = 32;

% Read in sample image
% I = im2double(imread('street1.jpg'));
% start = 150;
% I = I(start:start+N-1, start:start+N-1, 1);

% Case 1: Most basic case
% A is identity matrix
% b is the image itself
% A = eye(N*N);
% b = I(:);
% run_test(I, A, b, 3);

% Case 2: Blur by horizontal blurry filter
% convolve delta with this
% filter = [0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1];
% A = make_blurry_filter(N, filter);
% % imshow(A);
% % colormap(hot);
% b = A*I(:);
% lucy_debug(I, A, b, 30);

% Case 3: Noisy
% A is identity matrix
% b is the image itself
% use Sparse matrix to cut down on memory use
% A = sparse(make_blurry_filter(N, filter));
% var_gaus = 0.004;
% noisy = imnoise(I,"gaussian",0,var_gaus);
% b = A*noisy(:);
% lucy_debug(I, A, b, 30);

% Case 4: Synthetic OPT data
A = load('simulation_output_32x32.mat').A/10;
b = load('object.mat').x;
I = load('object.mat').I;
x = lucy_debug(cast(I, 'double'), A, b(:), 10);

%%
figure
imshow(act(A,x))
%% Predefined functions
function output = lucy(A, AT, b, iterations)
% Implementation of Lucy Richardson deconvolution
% Solves Ax = b
% assume x
one_vector = ones(size(b));
x = b;
% update rule for x is based on a multiplication
for i = 1:iterations
    % add eps to avoid division by 0
    x = AT * (b./(A*x + eps))./ (AT*one_vector) .* x; 
end
output = x;
end

function output = run_test(I,A, b, iterations)
% test function that does plots too
N = sqrt(length(A));
output = lucy(A, A' ,b, iterations );
output = reshape(output, [N,N]);

% display information
figure;
subplot(1,3,1);
imshow(I);
title('original');
subplot(1,3,2);
imshow(reshape(b, [N,N]));
title('measured');
subplot(1,3,3);
imshow(output);
title('LR output');
end

function output = lucy_debug(I,A,b,iterations)
N = sqrt(length(A));

% display information
figure;
subplot(1,3,1);
imshow(I);
title('A Original');
subplot(1,3,2);
imshow(reshape(b, [N,N]));
title('b Measured');

% Run LR update with plotting
one_vector = ones(size(b));
AT = A';
x = b;
% update rule for x is based on a multiplication
for idx = 1:iterations
    x = AT * (b./(A*x + eps))./ (AT*one_vector) .* x; 
    % plot info for this iteration
    subplot(1,3,3);
    imshow(reshape(x, [N,N]));
    title(sprintf('x LR output iteration %d ',idx ));
    pause(0.5)
end
output = x;
end

% helper functions
function b = act(A, x)
% A is transformation matrix with size (N*N,N*N)
% x is image with size (N,N)
% Does A*x and reshapes back to an image

N = sqrt(length(A));
b = reshape(A*x(:), [N,N]);
end

function A = make_blurry_filter(N, filter)
% creates A matrix for a horizontal blur, with size (N*N, N*N), based on
% blur given by the filter
% todo: Sparse filter generation
A = zeros(N*N);
I_deltas = eye(N); 

for image_row=1:N
    for image_col=1:N
        % Obtain point spread functions for each image_delta
        % Set columns in A matrix corresponding to the PSFs
        % PSF for this case is a horizontal blur, same for all rows in
        % image
        I_zeros = zeros(N);
        %I_zeros(image_row,:) = I_deltas(image_col,:); %conv(I_deltas(image_col,:), filter, 'same');
        I_zeros(image_row,:) = conv(I_deltas(image_col,:), filter, 'same');
        index = N*(image_col - 1) + image_row; % correct index for wrapping due to reshaping
        A(:,index) = I_zeros(:);
%         imshow(A);
    end
end

% % Blurry filter creation test cases
% % Test on impulse at certain location
% x = zeros(N,N);
% x(5,5) = 1;
% subplot(1,2,1);
% imshow(x);
% title('original');
% subplot(1,2,2);
% imshow(act(A,x))
% title('result');
% 
% % Test on image itself
% subplot(1,2,1);
% imshow(I);
% title('original');
% subplot(1,2,2);
% imshow(act(A,I))
% title('result');
% %
end
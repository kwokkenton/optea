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
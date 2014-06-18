%function driver

clear all
close all
format shortg

nDim_image = 50;
nDim_matrix = 8;

h_a = randn(nDim_matrix,nDim_matrix,nDim_image,nDim_image);
h_Q = zeros(nDim_matrix,nDim_matrix,nDim_image,nDim_image);
h_R = zeros(nDim_matrix,nDim_matrix,nDim_image,nDim_image);

%for i = 1:nDim_image
%    for j = 1:nDim_image
%        h_a(:,:,i,j) = [1,1,0;1,0,1;0,1,1];
%    end
%end

% transfer data to device
d_a  = gpuArray( h_a );
d_Q  = gpuArray( h_Q );
d_R  = gpuArray( h_R );

qrdptx = parallel.gpu.CUDAKernel('qrd.ptx', 'qrd.cu');
threadsPerBlock = 256;
npixel = nDim_image*nDim_image;
qrdptx.ThreadBlockSize=[threadsPerBlock, 1, 1];
%blocksPerGrid = (npixel + threadsPerBlock -1) / threadsPerBlock;
%blocksPerGrid = (npixel  * threadsPerBlock - 1) / threadsPerBlock;
blocksPerGrid = ceil(npixel/threadsPerBlock);
qrdptx.GridSize=[blocksPerGrid, 1, 1];
%qrdptx.GridSize=[ceil(blocksPerGrid), 1, 1];
%qrdptx.GridSize=[256, 1, 1];

[daout,dQout,dRout] = feval(qrdptx,d_a,d_Q,d_R,nDim_image,nDim_matrix);

%for i=1:nDim_image
%    for j=1:nDim_image
%        xtest(:,i,j)=h_A(:,:,i,j)\h_b(:,i,j);
%        xcheck = norm(xtest(:,i,j)-dxout(:,i,j));
%        if xcheck >= 1.0e-8 
%            xcheck
%        end
%    end
%end

%exit

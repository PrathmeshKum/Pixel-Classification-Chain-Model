
% Directed chain model program for pixel classification 

clc;
clear all;
close all;
warning('off', 'Images:initSize:adjustingMag');

%file directory
directory=char(pwd);

trainingImageDirectory = [directory '\trainingImages\'];
testingImageDirectory = [directory '\testingImages\'];
annotatedTrainingImageDirectory = [directory '\annotatedTrainingImages_Wenjin\'];
annotatedTestingImageDirectory = [directory '\annotatedTestingImages_Wenjin\'];

nDim = 32; %number of bins

%training process

tt= cputime;
Pr_x_given_w_equalsTo_1 = zeros(nDim,nDim,nDim);    % for unary costs 
Pr_x_given_w_equalsTo_0 = zeros(nDim,nDim,nDim);    % for unary costs 
trainingImageFiles = dir(trainingImageDirectory);
annotatedTrainingImageFiles = dir(annotatedTrainingImageDirectory);
N_face=0;

 for iFile = 3:size(trainingImageFiles,1)-1;  
     
        %load the image and facial image regions
        
        origIm=imread([trainingImageDirectory trainingImageFiles(iFile).name]);    
        bwMask = imread([annotatedTrainingImageDirectory annotatedTrainingImageFiles(iFile).name]);    
        
        %visualization and generate the mask indicating the facial regions
        
        origIm = imresize(origIm,0.125);
        bwMask = imresize(bwMask,0.125);
        [nrows,ncols,~]= size(origIm);
        showIm = origIm; showIm(bwMask) = 255;
        figure; imshow(showIm,[]);
        
        N_face = N_face + sum(bwMask(:));
        
        for iRow = 1:nrows;
            
            for iCol = 1:ncols;
                
                r=origIm(iRow,iCol,1);
                %r=(floor(r*1.0009))+1; % for converting to range: 1-256
                r=(floor(r*0.1216))+1; % 0.1216 factor for converting to 0-31 & +1 for range: 1-32
                g=origIm(iRow,iCol,2);
                %g=(floor(g*1.0009))+1;
                g=(floor(g*0.1216))+1;
                b=origIm(iRow,iCol,3);
                %b=(floor(b*1.0009))+1;
                b=(floor(b*0.1216))+1;
                
                if bwMask(iRow,iCol)==1;
                    
                    Pr_x_given_w_equalsTo_1(r,g,b) = Pr_x_given_w_equalsTo_1(r,g,b) + 1;
                
                else
                    
                    Pr_x_given_w_equalsTo_0(r,g,b) = Pr_x_given_w_equalsTo_0(r,g,b) + 1;
               
                end
            end
        end
        
 end
 
 % Normalizng
  
  Pr_x_given_w_equalsTo_1 = Pr_x_given_w_equalsTo_1/(sum(Pr_x_given_w_equalsTo_1(:)));
  Pr_x_given_w_equalsTo_0 = Pr_x_given_w_equalsTo_0/(sum(Pr_x_given_w_equalsTo_0(:)));
  
 disp(['traning: ' num2str(cputime-tt)]);
 
 %testing

testingFiles = dir(testingImageDirectory);
annotatedTestingImageFiles = dir(annotatedTestingImageDirectory);
file_num=1;

Pairwise_Cost_Matrix=[0.5, 0.5;0.5 0.5];  % Potts model matrix 

for iFile = 3:size(testingFiles,1)-1
    tt = cputime;
    
    %load the image and facial image regions
    
    origIm=imread([testingImageDirectory testingFiles(iFile).name]);    
    %detMask = imread([annotatedTestingImageDirectory annotatedTestingImageFiles(iFile).name]);
    
    origIm = imresize(origIm,0.125);
    [nrows, ncols,~] = size(origIm);
    detMask=zeros(nrows,ncols);
    
    % Inference
    
     for iRow = 1:nrows;
            
            for iCol = 1:ncols;
                
                r=origIm(iRow,iCol,1);
                %r=(floor(r*1.0009))+1; % for converting to range: 1-256  
                r=(floor(r*0.1216))+1; % 0.1216 factor for converting to 0-31 & +1 for range: 1-32
                g=origIm(iRow,iCol,2); 
                %g=(floor(g*1.0009))+1;
                g=(floor(g*0.1216))+1;
                b=origIm(iRow,iCol,3);
                %b=(floor(b*1.0009))+1;
                b=(floor(b*0.1216))+1;
                
                Row_Unary_Matrix(1,iCol)=-((Pr_x_given_w_equalsTo_1(r,g,b)));   % Unary cost matrix for row of image
                Row_Unary_Matrix(2,iCol)=-((Pr_x_given_w_equalsTo_0(r,g,b)));
                    
            end
            
            
            Final_Cost_Matrix=zeros(480,480);   % Matrix containing unary and pairwise costs at each nodes in the network
            count=1;
            count1=1;

            while count<=477;
    
                  Final_Cost_Matrix(count,count+2) = Pairwise_Cost_Matrix(1,1) + Row_Unary_Matrix(1,count1);
                  Final_Cost_Matrix(count,count+3) = Pairwise_Cost_Matrix(1,2) + Row_Unary_Matrix(1,count1);
                  Final_Cost_Matrix(count+1,count+2) = Pairwise_Cost_Matrix(2,1) + Row_Unary_Matrix(2,count1);
                  Final_Cost_Matrix(count+1,count+3) = Pairwise_Cost_Matrix(2,2) + + Row_Unary_Matrix(2,count1);
                  count=count+2;
                  count1=count1+1;
           end
            
           G = digraph(Final_Cost_Matrix);      % Creating direct chain model
           Tr = shortestpathtree(G,[1],[480]);  % Predicting pixel classes by optimal path
           W_test = Tr.Edges;                     
           W_test1=W_test.EndNodes(:,2);        % Matrix containing predicted class values
           
           for ii=1:size(W_test1,1);
               
               if mod(W_test1(ii,1),2) == 0;
                   
                  continue         % number is even (i.e it is estimated
                                   % as bg class)
                                        
               else
                   
                  detMask(iRow,ii)=1;    % number is odd (i.e it is estimated
                                           % as fc class)
               end
           end
                     
     end
       

    % Computing TP, FP,FN
    
    % for computation of ground truth (gtMask)
    
    gtMask = imread([annotatedTestingImageDirectory annotatedTestingImageFiles(iFile).name]);
    gtMask = imresize(gtMask,0.125);
    
    % True Positive (TP), False Positive (FP), False Negative (FN)
    
    detMask1=detMask;
    gtMask1=gtMask;
    tp=zeros(nrows,ncols);
    
    for iRow = 1:nrows;
        
        for iCol = 1:ncols;
            
            if detMask1(iRow,iCol)==1 && gtMask1(iRow,iCol)==1;
                
                tp(iRow,iCol)=1;
            
            else
                
                continue
            
            end
        end    
    end        
     
    TP(1,file_num)=(sum(sum(tp)));
    
    for iRow = 1:nrows;
        
        for iCol = 1:ncols;
            
            if detMask1(iRow,iCol)==1 && tp(iRow,iCol)==1;
                
                detMask1(iRow,iCol)=0;
            
            else
                
                continue
            
            end
        end    
    end
    
    fp=(sum(sum(detMask1)));
    FP(1,file_num)=fp;
    
    for iRow = 1:nrows;
        
        for iCol = 1:ncols;
            
            if gtMask1(iRow,iCol)==1 && tp(iRow,iCol)==1;
                
                gtMask1(iRow,iCol)=0;
            
            else
                
                continue
            
            end
        end    
    end
    
    fn=(sum(sum(gtMask1)));
    FN(1,file_num)=fn;
     
      
    disp([num2str(iFile-2) ' testing: ' num2str(cputime-tt)]);
   
    %some visualization
    
    showIm = origIm; showIm(nrows*ncols+find(detMask)) = 255;
    figure; imshow([origIm repmat(255*detMask,[1 1 3]) showIm],[]);
    
    file_num=file_num+1;
   
     
end


% Only for visualization. Not used in actual program

% s = [1 1 2 2 3 3 4 4];
% t = [3 4 3 4 5 6 5 6];
% weights = [0.1 0.9 0.9 0.1 0.9 0.1 0.1 0.9];G = digraph(s,t,weights);
% p = plot(G,'XData',x,'YData',y,'EdgeLabel',G.Edges.Weight);
% p = plot(G,'XData',x,'YData',y,'EdgeLabel',G.Edges.Weight);
% title('Plot for segment of directed chain model constructed for each row ');



% Computing the precision, recall and F-score
 
 precision=((sum(TP)/file_num)/((sum(TP)/file_num)+(sum(FP)/file_num)))*100; % In Percentage
 recall=((sum(TP)/file_num)/((sum(TP)/file_num)+(sum(FN)/file_num)))*100; % In Percentage
 f_score=((2*precision*recall)/(precision+recall));
 
 path=[directory '\chain_model_values.mat'];
 save(path);
 
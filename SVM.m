x=i;
Fx=o;
x  = x'; 
Fx = Fx';
[p, nVar] = size( x );
[ n_sample , n_var] = size( Fx );% number of class
%% Normalization 
rmin = 0; rmax = 1;
[xn  , xs ] = mapminmax( x'  , rmin , rmax);
[Fxn , Fxs] = mapminmax( Fx' , rmin , rmax);
xn  = xn'; Fxn = Fxn';
%%   Division of Data for Training, Testing
load( 'IndTrainTest.mat' )
nt = round( 0.8 * p );
JC =  19;%
trainInd = train( JC , 1 : nt);
testInd = TrainTest(  JC , nt + 1 : end);
 
SVR = fitrsvm( xn( trainInd , : ) , Fxn( trainInd , : ) ,...
                    'solver'         , 'SMO'   ,             'KernelFunction' , 'rbf'    ,   'KernelOffset'   , 0       ,  'KernelScale'    , 1    ,'BoxConstraint'  , 5  , 'CategoricalPredictors', CatePre);
%% Test SVR
pvn = predict( SVR , xn );
%% Unnormalized
pv = mapminmax( 'reverse' ,pvn , Fxs );

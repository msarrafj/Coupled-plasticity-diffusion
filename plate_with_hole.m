function out = coupled(formulation,hardening,anisotropy,sizeMesh,solid,coupling,Coupling_param)
%-------------------------------------------------------------------------------------------;
% formulation: <CG> or <NN>; hardening: <Model_I> or <model_II>                             ;
% anisotropy: <None> or <Low> or <High>                                                     ;
% sizeMesh: values <1> to <6>, one is the coarsest                                          ;
% solid: <el> for elastic; <ppl> for perfectly plastic, or  <elpl> for elastoplastic        ;
% Coupling: <None> for uncoupled, <One-way>, or <Two-way>                                   ;
% Coupling param: c_ref value in model_I and zeta value in model_II                         ;
%-------------------------------------------------------------------------------------------;

%--Load steps t=[0s,2.2s]--%
loadparam =[0 , 0.44:0.05:0.59 , 0.63:0.05:1.08 , 1.1:0.2:1.9 , 1.95:0.05:2.2];
ComsolTransportSolver = 'Off'; % <Off> to use matlab solver for transport problem

%--Importing the COMSOL class--%
import com.comsol.model.*
import com.comsol.model.util.*
model = ModelUtil.create('Model');

ModelUtil.showProgress(adname);
model.param.set('para', '0', 'Horizontal load parameter');
%--Anisotropy parameters--%
switch anisotropy
    case 'None'
        model.param.set('d_1', '1');
    case 'Low'
        model.param.set('d_1', '5');
    case 'High'
        model.param.set('d_1', '500');
end
model.param.set('d_2', '1');
%--Other material dataset
model.param.set('theta', 'pi/3');
model.param.set('lambda_0', '19444444444.4');
model.param.set('mu_0', '29166666666.7');
model.param.set('lambda_1', '-8.5e8');
model.param.set('mu_1', '-8.5e8');
model.param.set('phi_T', '2');
model.param.set('phi_S', '2');
model.param.set('E_ref', '0.001');
model.param.set('eta_T', '1');
model.param.set('eta_S', '1');
model.param.set('sigma0', '243e6', 'initial yield stress');
switch hardening
    case 'Model_I'
        model.param.set('c_ref', Coupling_param);
        model.param.set('E_t', '2.171e9', 'Isotropic tg modulus');
    case 'Model_II'
        model.param.set('zeta', Coupling_param);
        model.param.set('n', '5', 'work hardening');
end
model.variable.create('var1');
model.variable('var1').set('D_011', '(cos(theta))^2*d_1+(sin(theta))^2*d_2');
model.variable('var1').set('D_012', 'sin(theta)*cos(theta)*d_1-cos(theta)*sin(theta)*d_2');
model.variable('var1').set('D_022', '(cos(theta))^2*d_2+(sin(theta))^2*d_1');
model.variable('var1').set('D_T11', 'phi_T*D_011');
model.variable('var1').set('D_T12', 'phi_T*D_012');
model.variable('var1').set('D_T22', 'phi_T*D_022');
model.variable('var1').set('D_S11', 'phi_S*D_011');
model.variable('var1').set('D_S12', 'phi_S*D_012');
model.variable('var1').set('D_S22', 'phi_S*D_022');
model.variable('var1').set('trE', 'comp1.solid.el11+comp1.solid.el22');
model.variable('var1').set('DCoef_T', '(exp(eta_T*trE)-1)/(exp(eta_T*E_ref)-1)');
model.variable('var1').set('DCoef_S', '(exp(eta_S*trE)-1)/(exp(eta_S*E_ref)-1)');
model.variable('var1').set('E0', '(mu_0*(3*lambda_0+2*mu_0))/(lambda_0+mu_0)');

%--Define problem geometry--%
model.component.create('comp1', false);
model.component('comp1').geom.create('geom1', 2);
model.component('comp1').func.create('int1', 'Interpolation');
model.component('comp1').func('int1').set('funcname', 'loadfunc');
model.component('comp1').geom('geom1').lengthUnit('mm');
model.component('comp1').geom('geom1').repairTolType('relative');
model.component('comp1').geom('geom1').create('r1', 'Rectangle');
model.component('comp1').geom('geom1').feature('r1').set('size', [18 10]);
model.component('comp1').geom('geom1').create('c1', 'Circle');
model.component('comp1').geom('geom1').feature('c1').set('r', 5);
model.component('comp1').geom('geom1').create('c2', 'Circle');
model.component('comp1').geom('geom1').feature('c2').set('r', 7);
model.component('comp1').geom('geom1').create('dif1', 'Difference');
model.component('comp1').geom('geom1').feature('dif1').selection('input').set({'r1'});
model.component('comp1').geom('geom1').feature('dif1').selection('input2').set({'c1'});
model.component('comp1').geom('geom1').create('par1', 'Partition');
model.component('comp1').geom('geom1').feature('par1').selection('input').set({'dif1'});
model.component('comp1').geom('geom1').feature('par1').selection('tool').set({'c2'});
model.component('comp1').geom('geom1').feature('fin').set('repairtoltype', 'relative');
model.component('comp1').geom('geom1').create('mcd1', 'MeshControlDomains');
model.component('comp1').geom('geom1').feature('mcd1').selection('input').set('fin(1)', [1 2]);
model.component('comp1').geom('geom1').run;

%--Define mesh--%
model.component('comp1').meshd.create('mesh1');
model.component('comp1').mesh('mesh1').create('ftri1', 'FreeTri');
model.component('comp1').mesh('mesh1').create('ref1', 'Refine');
model.component('comp1').mesh('mesh1').feature('size').set('hauto', 8);
model.component('comp1').mesh('mesh1').feature('ref1').set('numrefine', sizeMesh);
model.component('comp1').mesh('mesh1').run;

%--Define deformation BVP--%
model.component('comp1').coordSystem.create('sys2', 'Cylindrical');
model.component('comp1').physics.create('solid', 'SolidMechanics', 'geom1');
model.component('comp1').physics('solid').feature('lemm1').create('plsty1', 'Plasticity', 2);
model.component('comp1').physics('solid').create('sym1', 'SymmetrySolid', 1);
model.component('comp1').physics('solid').feature('sym1').selection.set([1 3]);
model.component('comp1').physics('solid').create('bndl1', 'BoundaryLoad', 1);
model.component('comp1').physics('solid').feature('bndl1').selection.set([4]);
model.component('comp1').physics('solid').prop('ShapeProperty').set('order_displacement', 1);
model.component('comp1').physics('solid').prop('EquationForm').set('form', 'Stationary');
model.component('comp1').physics('solid').prop('Type2D').set('Type2D', 'PlaneStress');
model.component('comp1').physics('solid').prop('d').set('d', '10[mm]');
model.component('comp1').physics('solid').feature('lemm1').set('IsotropicOption', 'Lame');
model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
     .set('sigmags', 'sigma0');
model.component('comp1').physics('solid').feature('bndl1')...
     .set('FperArea', {'loadfunc(para)[MPa]'; '0'; '0'});
model.component('comp1').physics('solid').feature('lemm1').set('lambLame_mat', 'userdef');
model.component('comp1').physics('solid').feature('lemm1').set('muLame_mat', 'userdef');
model.component('comp1').physics('solid').feature('lemm1').set('rho_mat', 'userdef');
model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
     .set('sigmags_mat', 'userdef');
model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
     .set('Et_mat', 'userdef');
model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
     .set('e0_swi_mat', 'userdef');
model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
     .set('n_swi_mat', 'userdef');

%--Set deformation model--%
switch hardening
    case 'Model_I'
        model.component('comp1').physics('solid').feature('lemm1')...
             .set('lambLame', 'lambda_0+lambda_1*(comp1.c/c_ref)');
        model.component('comp1').physics('solid').feature('lemm1')...
             .set('muLame', 'mu_0+mu_1*(comp1.c/c_ref)');
        model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
             .set('Et', 'E_t');
    case 'Model_II'
        model.component('comp1').physics('solid').feature('lemm1').set('lambLame', 'lambda_0');
        model.component('comp1').physics('solid').feature('lemm1').set('muLame', 'mu_0');
        model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
             .set('sigmags', '(zeta*comp1.c+1)*sigma0');
        model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
             .set('IsotropicHardeningModel', 'Model_II');
        model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
             .set('e0_swi', 'sigma0/E0');
        model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
             .set('n_swi', 'n');
end

if ( strcmp( solid ,'ppl') )
    fprintf('Perfectly_plastic model is enforced for this problem!\n'); 
model.component('comp1').physics('solid').feature('lemm1').feature('plsty1')...
     .set('IsotropicHardeningModel', 'PerfectlyPlastic');
end

if ( strcmp( solid ,'el') )
    fprintf('Elasticity model is enforced for this problem!\n'); 
    model.component('comp1').physics('solid').feature('lemm1').feature('plsty1').active(false);
end

%--Set diffusion BVP--%
model.component('comp1').physics.create('tds', 'DilutedSpecies', 'geom1');
model.component('comp1').physics('tds').create('nflx2', 'NoFlux', 1);
model.component('comp1').physics('tds').feature('nflx2').selection.set([4]);
model.component('comp1').physics('tds').create('sym1', 'Symmetry', 1);
model.component('comp1').physics('tds').feature('sym1').selection.set([1 3]);
model.component('comp1').physics('tds').create('conc1', 'Concentration', 1);
model.component('comp1').physics('tds').feature('conc1').selection.set([2]);
model.component('comp1').physics('tds').create('conc2', 'Concentration', 1);
model.component('comp1').physics('tds').feature('conc2').selection.set([5]);
model.component('comp1').physics('tds').prop('ShapeProperty')...
     .set('boundaryFlux_concentration', false);
model.component('comp1').physics('tds').prop('EquationForm').set('form', 'Transient');
model.component('comp1').physics('tds').prop('TransportMechanism').set('Convection', false);
model.component('comp1').physics('tds').prop('MassConsistentStabilization')...
     .set('massStreamlineDiffusion', false);
model.component('comp1').physics('tds').prop('MassConsistentStabilization')...
     .set('massCrosswindDiffusion', false);
model.component('comp1').physics('tds').feature('conc1').set('species', true);
model.component('comp1').physics('tds').feature('conc2').set('species', true);
model.component('comp1').physics('tds').feature('conc2').set('c0', 1);

%--Set one-way vs two-way coupling strategy--%
if ( strcmp( coupling ,'One_way') )
    fprintf('Considered one-way coupling!\n'); 
    model.component('comp1').physics('tds').feature('cdm1').set('D_c', {'D_011'; 'D_012';...
        '0'; 'D_012'; 'D_022'; '0'; '0'; '0'; '0'});
elseif ( strcmp( coupling ,'Two_way') )
    fprintf('Considered two-way coupling!\n'); 
    model.component('comp1').physics('tds').feature('cdm1').set('D_c',...
        {'D_011+(D_T11-D_011)*DCoef_T+(D_S11-D_011)*DCoef_S';...
            'D_012+(D_T12-D_012)*DCoef_T+(D_S12-D_012)*DCoef_S'; '0';...
            'D_012+(D_T12-D_012)*DCoef_T+(D_S12-D_012)*DCoef_S';...
            'D_022+(D_T22-D_022)*DCoef_T+(D_S22-D_022)*DCoef_S'; '0'; '0'; '0'; '0'});
elseif (strcmp(coupling, 'None') )
    fprintf('Considered No way coupled problem!\n')
else
    fprintf('Incorrect coupling option selected!\n'); 
    return
end

%--Initialize deformation solver--%
model.study.create('std1');
model.study('std1').create('stat', 'Stationary');
model.study('std1').feature('stat').set('activate', {'solid' 'on' 'tds' 'off'});
model.study('std1').label('deformation');
model.study('std1').feature('stat').set('useinitsol', true);
model.study('std1').feature('stat').set('usesol', true);
model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').attach('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
model.sol('sol1').feature('s1').create('i1', 'Iterative');
model.sol('sol1').feature('s1').feature('i1').create('mg1', 'Multigrid');
model.sol('sol1').feature('s1').feature('i1').feature.remove('ilDef');
model.sol('sol1').feature('s1').feature.remove('dDef');
model.sol('sol1').feature('s1').feature.remove('fcDef');

%--Set solver settings for deformation--%
model.sol('sol1').attach('std1');
model.sol('sol1').feature('v1').feature('comp1_c').set('out', false);
model.sol('sol1').feature('s1').set('stol', '0.000001'); % solver tolerance
model.sol('sol1').feature('s1').set('nonlin', false);
model.sol('sol1').feature('s1').set('linpmethod', 'sol');
model.sol('sol1').feature('s1').set('linpsol', 'sol1');
model.sol('sol1').feature('s1').set('solnum', 'last');
model.sol('sol1').feature('s1').feature('aDef').set('convinfo', 'detailed');
model.sol('sol1').feature('s1').feature('i1').set('linsolver', 'precond');
model.sol('sol1').feature('s1').feature('i1').set('errorchk', true);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('prefun', 'saamg');
model.sol('sol1').runAll; % obtained solution field at t=0s 
model.study('std1').label('deformation');
model.study('std1').feature('stat').set('useinitsol', true);
model.study('std1').feature('stat').set('initstudy', 'std1');
model.study('std1').feature('stat').set('solnum', 'last');
model.study('std1').feature('stat').set('usesol', true);
if ( strcmp( coupling ,'One_way') ) || ( strcmp( coupling ,'Two_way') )
    model.study('std1').feature('stat').set('notsolmethod', 'sol');
end
model.study('std1').feature('stat').set('notstudy', 'std2');
model.study('std1').feature('stat').set('notsolnum', 'last');
model.sol('sol1').attach('std1');
model.sol('sol1').feature('v1').set('control', 'user');
model.sol('sol1').feature('v1').set('initmethod', 'sol');
model.sol('sol1').feature('v1').set('initsol', 'sol1');
model.sol('sol1').feature('v1').set('solnum', 'last');
if ( strcmp( coupling ,'One_way') ) || ( strcmp( coupling ,'Two_way') )
    model.sol('sol1').feature('v1').set('notsolmethod', 'sol');
end
model.sol('sol1').feature('v1').set('notsol', 'sol2');
model.sol('sol1').feature('v1').set('notsolnum', 'last');

%--initialize diffusion solver--%
model.study.create('std2');
model.study('std2').create('stat', 'Stationary');
model.study('std2').feature('stat').set('activate', {'solid' 'off' 'tds' 'on'});
model.study('std2').label('diffusion');
model.study('std2').feature('stat').set('useinitsol', true);
model.study('std2').feature('stat').set('usesol', true);
model.study('std2').feature('stat').set('notsolmethod', 'sol');
model.study('std2').feature('stat').set('notstudy', 'std1');
model.study('std2').feature('stat').set('notsolnum', 'last');
model.sol.create('sol2');
model.sol('sol2').study('std2');
model.sol('sol2').attach('std2');
model.sol('sol2').create('st1', 'StudyStep');
model.sol('sol2').create('v1', 'Variables');
model.sol('sol2').create('s1', 'Stationary');
model.sol('sol2').feature('s1').create('fc1', 'FullyCoupled');
model.sol('sol2').feature('s1').create('i1', 'Iterative');
model.sol('sol2').feature('s1').feature.remove('fcDef');

%--Set solver settings for diffusion--%
model.sol('sol2').attach('std2');
model.sol('sol2').feature('v1').set('notsolmethod', 'sol');
model.sol('sol2').feature('v1').set('notsol', 'sol1');
model.sol('sol2').feature('v1').set('notsolnum', 'last');
model.sol('sol2').feature('v1').feature('comp1_u').set('out', false);
model.sol('sol2').feature('s1').set('stol', '0.00001'); % tolerance for solver
model.sol('sol2').feature('s1').set('nonlin', false);
model.sol('sol2').feature('s1').set('linpmethod', 'sol');
model.sol('sol2').feature('s1').set('linpsol', 'sol2');
model.sol('sol2').feature('s1').set('solnum', 'auto');
model.sol('sol2').feature('s1').feature('aDef').set('convinfo', 'detailed');
model.sol('sol2').feature('s1').feature('fc1').set('initstep', 0.01);
model.sol('sol2').feature('s1').feature('fc1').set('minstep', 1.0E-6);
model.sol('sol2').feature('s1').feature('fc1').set('maxiter', 500);
model.sol('sol2').feature('s1').feature('i1').set('irestol', '0.0000001');
model.sol('sol2').feature('s1').feature('i1').set('maxlinit', 1000000);
model.sol('sol2').feature('s1').feature('i1').set('errorchk', true);
model.sol('sol2').feature('s1').feature('i1').feature('ilDef').set('droptol', 1.0E-5);
model.sol('sol2').runAll; % obtained solution field at t=0s
model.study('std2').label('diffusion');
model.study('std2').feature('stat').set('useinitsol', true);
model.study('std2').feature('stat').set('initmethod', 'sol');
model.study('std2').feature('stat').set('initstudy', 'std2');
model.study('std2').feature('stat').set('solnum', 'last');
model.study('std2').feature('stat').set('usesol', true);
model.study('std2').feature('stat').set('notsolmethod', 'sol');
model.study('std2').feature('stat').set('notstudy', 'std1');
model.study('std2').feature('stat').set('notsolnum', 'last');
model.sol('sol2').attach('std2');
model.sol('sol2').feature('v1').set('control', 'user');
model.sol('sol2').feature('v1').set('initmethod', 'sol');
model.sol('sol2').feature('v1').set('initsol', 'sol2');
model.sol('sol2').feature('v1').set('solnum', 'last');
model.sol('sol2').feature('v1').set('notsolmethod', 'sol');
model.sol('sol2').feature('v1').set('notsol', 'sol1');
model.sol('sol2').feature('v1').set('notsolnum', 'last');

%--Map COMSOL dofs to MATLAB dofs--%
info = mphxmeshinfo(model);
nodofs = info.ndofs;
idx_dofs = info.elements.tri.dofs(:,:);
idx_dofs = idx_dofs +1;
info.dofs.dofnames;
idx_names = info.dofs.nameinds(idx_dofs);
idx_dofnames = find(strcmp(info.dofs.dofnames,'comp1.c'))-1;
idx = find(idx_names==idx_dofnames);
c_index_global = idx_dofs(idx);
info = mphxmeshinfo(model,'soltag','sol2','studysteptag','v1');
nodofs_c = info.ndofs;
idx_dofs_c = info.elements.tri.dofs(:,:);
idx_dofs_c = idx_dofs_c + 1;
CMapped_index_matrix=zeros(3*size(info.elements.tri.dofs(:,:),2),2);
CMapped_index_matrix(:,1) = Vector(idx_dofs_c);
CMapped_index_matrix(:,2) = c_index_global;
CMap = unique(CMapped_index_matrix, 'rows');

%--Solve staggered problem--%
count = 1;
for i = loadparam
    model.param.set('para', i , 'Horizontal load parameter');

    %--Solve deformation problem using COMSOL built-in solver--%
    model.sol('sol1').runAll;
    U1 = mphgetu(model,'soltag','sol1');

    %--Solve diffusion problemr-%
    %--Assembly phase--%
    U2 = mphgetu(model,'soltag','sol2');
    MAc = mphmatrix(model ,'sol2', ...
    'Out', {'Kc','Lc','Null','ud','uscale','Nullf'},...
    'initmethod','init','Study','std2');
    udc = MAc.ud;
    Nullc = full(MAc.Null);
    Nullfc = full(MAc.Nullf);
    k_fp_u_pc=MAc.Kc*MAc.Nullf'*MAc.ud;

    %--Solver phase--%
    switch formulation
        case 'NN' %--Non-negative formulation--%
            guess = ones(size(MAc.Lc,1),1);
            l_b = -eps * ones(size(MAc.Lc,1),1);
            u_b = ones(size(MAc.Lc,1),1);
            QP_algorithm = 'trust-region-reflective';
            options = optimset('Algorithm',QP_algorithm,...
                      'MaxIter',75000,'TolX',100*eps,'TolFun',100*eps);
            %--Call MATLAB built-in quadprog prog solver--% 
            MAc.Kc = 0.5 * (MAc.Kc + MAc.Kc');
            [Unn_F,fval,exitflag,output,lambda] = quadprog(MAc.Kc,-1.0 *...
            (MAc.Lc - k_fp_u_pc),[],[],[],[],l_b,u_b,guess,options);
            Unn_final0 = MAc.Null*Unn_F;
            Unn_final = (Unn_final0+MAc.ud).*MAc.uscale;
            U2_new_NN = U2;
            for i=1:length(CMap)
            U2_new_NN(CMap(i,2))= Unn_final(i,1);
            end
            model.sol('sol2').clearSolution();
            model.sol('sol2').setU(U2_new_NN);
            model.sol('sol2').createSolution();
            output
        case 'CG' %--Continuous Galerkin formulation--%
            tol = 1e-6; maxit = 10000000;
            [L,U] = ilu(MAc.Kc, struct('type','ilutp','droptol',1e-6));
            restart = 50;
            %--Call MATLAB built-in GMRES soler--%
            [Ucu,exitflag,relres,iter_conc,resvec] = gmres(MAc.Kc,MAc.Lc,restart,tol,maxit,L,U);
            U0c = MAc.Null*Ucu+MAc.ud;
            Ucg_matlabc = (U0c).*MAc.uscale;
            U2_new_CG = U2;
            for i=1:length(CMap)
            U2_new_CG(CMap(i,2))= Ucg_matlabc(i,1);
            end
            model.sol('sol2').clearSolution();
            model.sol('sol2').setU(U2_new_CG);
            model.sol('sol2').createSolution();
            vec1 = iter_conc';
            vec2 = resvec(2:(length(resvec)));
            format longEng
            GMRES  = [vec1 vec2]
    end
count = count+1;
end
out = model;

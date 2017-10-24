clc
clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Function gaussianmixture %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test de la fonction gaussian mixture pour deux classes
% Paramètres
pi = [2/3;1/3];
mu = [1/2,1/3;
       -1/5,-2];
sigma = [eye(2,2),eye(2,2)];
n = 1000;
% Lancement de la fonction gaussian mixture
[X, classes] = gaussianmixture(n,pi,mu,sigma);
% Affichage des résultats
figure
for i=1:n
    if classes(i)==1
        p1 = plot(X(i,1),X(i,2),'rx');
        hold on
    else
        p2 = plot(X(i,1),X(i,2),'go');
        hold on
    end
end
legend([p1,p2],'Loi 1','Loi 2')
title('Echantillon dun modèle de mélange avec K=2')

%% Test de la fonction gaussian mixture pour cinq classes
% Paramètres
pi = [1/10;2/10;2/5;1/5;1/10];
mu = [1,1;
       3,4;
       -1,3;
       -1,-1
       -5,6];
sigma = [eye(2,2),eye(2,2),eye(2,2),eye(2,2),eye(2,2)];
n = 5000;
% Lancement de la fonction gaussian mixture
[X, classes] = gaussianmixture(n,pi,mu,sigma);
% Affichage des résultats
figure
for i=1:n
    if classes(i)==1
        p1 = plot(X(i,1),X(i,2),'rx');
        hold on
    elseif classes(i)==2
        p2 = plot(X(i,1),X(i,2),'go');
        hold on
    elseif classes(i)==3
        p3 = plot(X(i,1),X(i,2),'y+');
        hold on
    elseif classes(i)==4
        p4 = plot(X(i,1),X(i,2),'m*');
        hold on
    else
        p5 = plot(X(i,1),X(i,2),'b^');
        hold on
    end
end
legend([p1,p2,p3,p4,p5],'Loi 1','Loi 2','Loi 3','Loi 4','loi 5')
title('Echantillon dun modèle de mélange avec K=5')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fonction algorithme_EM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Lancement de l'algorithme EM
K = 5
tolerance = 10^(-16);
[mu_estime,sigma_estime,pi_estime,classe_estime,~] = algorithme_EM(X,K,tolerance);
% On associe les classes de K-means aux bonnes classes
p = perms(1:K);
erreurs = zeros(size(p,1),1);
estimated_classes_changed_matrix = zeros(n,size(p,1));
for i=1:size(p,1)
    estimated_classes_changed = zeros(size(classe_estime));
    for j=1:K
        estimated_classes_changed = estimated_classes_changed + (classe_estime==j)*p(i,j);
    end
    estimated_classes_changed_matrix(:,i) = estimated_classes_changed;
    erreurs(i) = sum(estimated_classes_changed~=classes);
end
[val,ind] = min(erreurs);
disp(['Lerreur sur la base est alors de ',num2str(val/n*100,10^(-2)),'%'])
% Affichage du résultat
figure
for i=1:n
    if estimated_classes_changed_matrix(i,ind)==1
        p1 = plot(X(i,1),X(i,2),'rx');
        hold on
    else
        p2 = plot(X(i,1),X(i,2),'b^');
        hold on
    end
end
legend([p1,p2],'Loi 1','Loi 2')
title('Resultat obtenu par lalgorithme EM')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Nombre optimal de cluster %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Paramètres
% maxK represente le nombre maximal de cluster
maxK = 5;
% logL est un vecteur contenant les vraisemblances logarithmiques à determiner à chaque iteration
logL = zeros(maxK,1);
% numParam est le vecteur contenant les nombres de parametres à determiner à chaque iteration
numParam = zeros(maxK,1);
% numObs est le nombre d'observations
numObs = n;
% Dimension du problème
dimension = 2;
% Tolérance pour accélerer le processus
tolerance = 10^(-3);

%% Iteration sur les valeurs de K
for k=1:maxK
    [~,~,~,~,vraisemblance]=algorithme_EM(X,k,tolerance);
    logL(k) = vraisemblance;
    numParam(k) = k+dimension*k+dimension^2*k;
end

%% AIC et BIC
[aic,bic]=aicbic(logL,numParam,numObs);
[val_aic,ind_aic] = min(aic);
disp(['Le critere AIC indique K=',num2str(ind_aic),' classe(s)'])
[val_bic,ind_bic] = min(bic);
disp(['Le critere BIC indique K=',num2str(ind_bic),' classe(s)'])


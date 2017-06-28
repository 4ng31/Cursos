function []=svcplot(NET, x, y, alpha, b);
% AFFICHAGE GRAPHIQUE
%
%   Syntaxe :	svcplot(NET, x, y, alpha, b);
%
%   ENTREE(s)
%       NET     structure de donn�e
%       x       matrice o� chaque ligne est un individu
%       y       vecteur colonne indiquant la classe d'appartenance (y(i) = +/-1)
%       alpha   multiplicateurs de Lagrange
%       b       biais
%
%   Voir aussi : CREATENET
%
% Derni�re mise � jour : 19/07/04 (POTHIN JB)

if ((nargin ~= 5) & (nargin ~= 6))
    help svcplot
else

    q = 50;           % pr�cision de la grille pour le calcul de la fronti�re

    [n,p] = size(x);
    if (p ~= 2)
        error('Les individus ne peuvent pas etre repr�sent�s dans le plan')
    end

    figure
    hold on
   
    % RECHERCHE DES INDIVIDUS
    C = NET.C;
    alptol = NET.alptol;

    i1 = find(y == -1);     % indice individu classe -1  
    i2 = find(y == +1);     % indice individu classe +1    
    sv = alpha > alptol;    % 1 si vecteur support, 0 sinon  
    y = y.*sv;              % on r�alise un masque pour ne conserver que les vecteurs supports 
    isv1 = find(y == -1);   % indice des vecteurs supports classe -1
    isv2 = find(y == +1);   % indice des vecteurs supports classe +1
    isv_m = find(sv & (alpha < C-alptol)); % indice des vecteurs supports sur la marge   
    
    % AFFICHAGE DE LA CLASSE Y = -1
	plot(x(i1,1), x(i1,2),'vr');
    

    % AFFICHAGE DE LA CLASSE Y = +1
	plot(x(i2,1), x(i2,2),'hb')
    
    
    % MODIFICATION DES AXES
    xmin = min(x(:,1)) - 1;
    xmax = max(x(:,1)) + 1;
    ymin = min(x(:,2)) - 1;
    ymax = max(x(:,2)) + 1;
    axis([xmin xmax ymin ymax]);
    
    % TRACER DE LA FRONTIERE DE DISCRIMINATION     
    % cr�ation de la grille d'�chantillonnage
    delta_x = (xmax-xmin)/q; % pas de quantification pour l'axe des abscisses
    delta_y = (ymax-ymin)/q; % pas de quantification pour l'axe des ordonn�es

    surface=ones(q,q);
    xx = [];    yy = [];
    xdess=[];
    
    for j=1:q
        aa = xmin + j*delta_x;
        bb = ymin + j*delta_y;
        xx = [xx; aa];
        yy = [yy, bb];
        for i=1:q
            aa = xmin + i*delta_x;
            xdess = [xdess; [aa bb]];
        end
    end

    d = mysvc(NET, x, y, xdess, alpha, b); % classification des individus dans xdess
    surface = d;
    size(d);
    surface = reshape(surface,q,q);
    contour(xx,yy,surface',[0 0],'k'); % representation du discriminateur    
    
    if strcmp(lower(NET.displ), 'few') | strcmp(lower(NET.displ), 'all')
        % AFFICHAGE DES VECTEURS SUPPORTS DE LA CLASSE Y = -1
        plot(x(isv1,1), x(isv1,2), 'or')

        % AFFICHAGE DES VECTEURS SUPPORTS DE LA CLASSE Y = +1
        plot(x(isv2,1), x(isv2,2), 'ob')

        % AFFICHAGE DES VECTEURS SUPPORTS SUR LA MARGE    
        plot(x(isv_m,1), x(isv_m,2), 'ok')
        
        dsup = mysvc(NET, x, y, xdess, alpha, b+1);
        dinf = mysvc(NET, x, y, xdess, alpha, b-1);
        
        surface = dsup;
        surface = reshape(surface,q,q);
        contour(xx,yy,surface',[0 0],':k'); % representation de la marge

        surface = dinf;
        surface = reshape(surface,q,q);
        contour(xx,yy,surface',[0 0],':k'); % representation de la marge    
          
        % affichage de la surface en 3D
       figure
       surf(xx,yy,surface)
       colormap('gray')
       shading('interp');
    end

    hold off
end
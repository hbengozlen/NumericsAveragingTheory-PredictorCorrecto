function theta_phy=getTheta0(Nx,Ny,cutoff,Theta0_amplitude)

% 1. Zufälliges Spektrum im Frequenzraum erzeugen
% Wir starten mit komplexen Zufallszahlen
theta_four = (rand(Ny, Nx) - 0.5) + 0*1i * (rand(Ny, Nx) - 0.5);

% 2. Tiefpass-Filter: Nur kleine Moden behalten
% In der Standard-FFT sitzen die niedrigen Frequenzen in den Ecken.
% Einfacher ist es mit fftshift:
theta_four_shifted = fftshift(theta_four);

% Definiere den Radius der erlaubten Moden (Cut-off)
centerx = floor(Nx/2) + 1;
centery = floor(Ny/2) + 1;

for i = 1:Ny
    for j = 1:Nx
        % Berechne Distanz zum Zentrum (Gleichanteil/niedrige Frequenzen)
        if sqrt((i-centery)^2 + (j-centerx)^2) > cutoff
            theta_four_shifted(i,j) = 0; % Hohe Moden löschen
        end
    end
end

% 3. Zurücktransformieren
% Zurückshiften nicht vergessen!
theta_phy = real(ifft2(ifftshift(theta_four_shifted)));
theta_phy=Theta0_amplitude*theta_phy./max(abs(theta_phy(:)));

end
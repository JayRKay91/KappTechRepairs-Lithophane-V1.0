; === Litho Mesh Studio - Inno Setup Installer Script V1.0 ===

#define MyAppName "Litho Mesh Studio"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Kapp Tech Repairs"
#define MyAppExeName "LithoMeshStudio.exe"

[Setup]
; Unique App ID (generated for V1)
AppId={{D3F28190-7F8A-42B1-9B5F-B40E7D2E551A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Output folder and installer binary name
OutputDir=installer_output
OutputBaseFilename=LithoMeshStudio_Setup_v1.0.0
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application executable
Source: "dist\LithoMeshStudio\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
; All supporting binaries, DLLs, and customtkinter data
Source: "dist\LithoMeshStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Option to launch the program immediately after install completes
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
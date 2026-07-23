Unicode true
RequestExecutionLevel user

!include MUI2.nsh

!define APP_NAME "城市夜景AIGC图像检测系统"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "城市夜景AIGC图像检测系统"
!define APP_EXE "AIGC_Detector.exe"
!define APP_ID "AIGC_Detector_NightCity"

Name "${APP_NAME}"
Caption "${APP_NAME} 安装程序"
OutFile "installer\AIGC_Detector_Setup.exe"
InstallDir "$LOCALAPPDATA\Programs\${APP_NAME}"
InstallDirRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "InstallLocation"

!define MUI_ICON "data\img\tup_1.ico"
!define MUI_UNICON "data\img\tup_1.ico"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "欢迎安装 ${APP_NAME}"
!define MUI_WELCOMEPAGE_TEXT "本安装程序会将软件安装到当前用户目录，不需要管理员权限。$\r$\n$\r$\n请确保安装包旁边保留 payload 文件夹，否则无法完成安装。"
!define MUI_DIRECTORYPAGE_TEXT_TOP "请选择软件安装位置。推荐使用默认路径。"
!define MUI_INSTFILESPAGE_FINISHHEADER_TEXT "安装完成"
!define MUI_INSTFILESPAGE_FINISHHEADER_SUBTEXT "${APP_NAME} 已安装到你的电脑。"
!define MUI_FINISHPAGE_TITLE "安装完成"
!define MUI_FINISHPAGE_TEXT "${APP_NAME} 已成功安装。你可以通过桌面快捷方式或开始菜单启动软件。"
!define MUI_FINISHPAGE_RUN "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "立即运行 ${APP_NAME}"
!define MUI_UNCONFIRMPAGE_TEXT_TOP "即将卸载 ${APP_NAME}。"

ShowInstDetails show
ShowUninstDetails show

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "SimpChinese"

Section "安装主程序"
    SetShellVarContext current
    SetOutPath "$INSTDIR"

    IfFileExists "$EXEDIR\payload\AIGC_Detector\${APP_EXE}" 0 payload_missing

    DetailPrint "正在复制程序文件和模型权重，这可能需要几分钟..."
    nsExec::ExecToLog '"$SYSDIR\robocopy.exe" "$EXEDIR\payload\AIGC_Detector" "$INSTDIR" /E /COPY:DAT /R:2 /W:1'
    Pop $0
    IntCmp $0 8 copy_failed copy_ok copy_ok

payload_missing:
    MessageBox MB_ICONSTOP "安装文件不完整：缺少 $EXEDIR\payload\AIGC_Detector。请确认 payload 文件夹和安装程序放在同一目录。"
    Abort

copy_failed:
    MessageBox MB_ICONSTOP "复制程序文件失败。Robocopy返回码：$0"
    Abort

copy_ok:
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\${APP_EXE}" 0
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\${APP_EXE}" 0

    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "Publisher" "${APP_PUBLISHER}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "DisplayIcon" "$INSTDIR\${APP_EXE}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "NoModify" 1
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" "NoRepair" 1
SectionEnd

Section "un.卸载"
    SetShellVarContext current
    Delete "$DESKTOP\${APP_NAME}.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
    RMDir "$SMPROGRAMS\${APP_NAME}"

    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}"

    RMDir /r "$INSTDIR"
SectionEnd

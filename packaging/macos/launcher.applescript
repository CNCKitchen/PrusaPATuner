-- Prusa PA Tuner -- macOS launcher applet.
--
-- Built into "Prusa PA Tuner.app" by scripts/make_macos_app.sh, which replaces the
-- two @@...@@ placeholders below with paths on this machine.
--
--   * launch (double-click)  -> start the local server if it is not running yet,
--                               then open the web UI in the default browser
--   * click the Dock icon    -> open the web UI again
--   * quit (Cmd+Q)           -> stop the server
--
-- Server output goes to ~/Library/Logs/PrusaPATuner.log.

property tunerExe : "@@TUNER_EXE@@"
property workDir : "@@WORK_DIR@@"
property uiURL : "http://127.0.0.1:8765/"

on isUp()
	try
		do shell script "/usr/bin/curl -s -o /dev/null -m 2 " & quoted form of (uiURL & "api/version")
		return true
	on error
		return false
	end try
end isUp

on openUI()
	-- `open` (rather than AppleScript's `open location`) always honours the
	-- user's default browser.
	do shell script "/usr/bin/open " & quoted form of uiURL
end openUI

on run
	if not isUp() then
		try
			-- Keep these as separate `;` statements: chaining them with `&&` in
			-- front of the trailing `&` backgrounds the whole list in a subshell
			-- that keeps do-shell-script's stdout open, and the applet hangs.
			do shell script "mkdir -p \"$HOME/Library/Logs\"; cd " & quoted form of workDir & " || exit 1; nohup " & quoted form of tunerExe & " --no-browser > \"$HOME/Library/Logs/PrusaPATuner.log\" 2>&1 < /dev/null &"
		on error errMsg
			display dialog "Prusa PA Tuner could not be started:" & return & errMsg buttons {"OK"} default button 1 with icon stop
			quit
			return
		end try
		set ok to false
		repeat 40 times
			if isUp() then
				set ok to true
				exit repeat
			end if
			delay 0.5
		end repeat
		if not ok then
			display dialog "Prusa PA Tuner did not come up. See ~/Library/Logs/PrusaPATuner.log for details." buttons {"OK"} default button 1 with icon stop
			quit
			return
		end if
	end if
	openUI()
end run

on reopen
	openUI()
end reopen

on idle
	return 60
end idle

on quit
	try
		do shell script "/usr/bin/pkill -f " & quoted form of tunerExe & "; true"
	end try
	continue quit
end quit

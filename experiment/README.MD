 * **_Crucial_ tip 1: when I run the experiments, I use `ctrl+r` to find the old command that I need, then I just slighty edit the old command where necessary (usually the subject and/or session). I almost never type out the complete commands myself.**
 * **_Crucial_ tip 2: The scripts have a protection against overwriting log files that are already there. When logfiles already exist, the script will ask whether you are sure about overwriting them. If you have to continue, the correct answer is `y` (so not `yes`). I think the best strategy is to never use session 1/2/3 for testing scripts, but use session 99 or something.**

Some more tips:

 * Marius knows where the key + locker is in which there is the money and necessary forms.
 * The forms can also be found on the "transfer computer" in the scanning room. You can print them from there. I think it is either in `Desktop/Gilles` or `Documents/Gilles`. Karl would know.

# Session 1 (no TMS)

## Instruction phase
 * Use the stimulus computer in the *Single Subject Room*.
 * Open Powershell
   * Press the start button and type "power". When you see powershell selected, press enter.
  * Activate the virtual environment:
    * `Documents\Gilles\virtualenv\Scripts\activate`
  * Go to the experiment folder `cd Documents\Gilles\tms_risk\experiment`
 * Run the **instruction script**: `python instruct.py --settings macbook`
 * When the subject arrives: give them first all the five forms:
   * Informed consent (only text)
   * Informed consent (form to sign)
   * Information about MRI
   * MRI checklist
   * TMS chekclist
  * After they have filled in all the forms and you checked the screening forms. 
  * Make them change into the green garments.
  * Let them read the instructions sheet. After they have done that, they can do the instruction task on the computer.

## Task phase
 .....

# Session 2/3 (TMS)

## Form phase
 * When the subject arrives: give them first all these four forms:
   * Informed consent (only text)
   * Informed consent (form to sign)
   * MRI checklist
   * TMS checklist
  * After they have filled in all the forms and you checked the screening forms, make them change into the green garments.
  * Now they can go to the TMS preparation room.

## TMS phase
 * Marius takes care of this.
 
## Task phase

### Before subject enters
Before the subject enters the scanning room (coming from TMS prep), prepare a bit:
 * Make sure the power of the projector and eye tracking computer are on (_do this before booting up the Eyelink an Stimulus computer_). When you walk into the scanner room there is a door at the end of the room on the right. There you find a little techincal room. You can see a plug hanging next to the socket. This is from the eye tracker. Put it in the socket. There is also a remote for the projector. Use the power button on the top left of the remote to turn on the projector.
 * *Only now* you can turn on both the Eyelink and stimulus computer. The Eyelink computer has a power button on its front. For the stimulus computer, the power button is on top of the case, close to the face of the case.
 * After booting up, ppen Powershell on the stimulus computer:
   * Press the start button and type "power". When you see Powershell selected, press enter.
 * Activate the virtual environment: `Documents\Gilles\risk_experiment\virtualenv\Scripts\activate` (*Maike*, please check this. You could also use the identical virtualenv in our stress study `C:\Users\Econ_Experimenter\Documents\SNS\Gilles\stress_risk\experiment\virtualenv\risk\Scripts\activate`)
    * Maike: what worked was ` C:\Users\Econ_Experimenter\Documents\SNS\Gilles\risk\virtualenv\risk\Scripts\activate`
 * Make a new trial design for the subject, so we have randomized order across sessions:
   * `cd Documents\SNS\Gilles\tms_risk\experiment`
   * `python make_trial_design.py <subject> 1 1` (Session 1 corresponds to the session on which the calibration was done, not the current session)

### When subject enters
* When Marius puts the subject in the scanner the first time, you can open the `Eye (monitor 1)` (or something like that)-script on the desktop. This will show a eye calibrations cript on the scanner display.
* Press `enter` to show the input to the camera. Marius will know what to do next.
* Once the eye camera is okay, you can quit the calibration script by pressing `escape`
* You will now make the first survey scan, before stimulation.
* Make sure that the task script is ready to run _before stimulation of the subject_.
* It is a very small bash script straight from the terminal that looks like this (just `ctrl+r` it):
```
for ($run = 1 ; $run -lt 7 ; $run++){    
python task.py <SUBJECT> <SESSION> $run --settings 3t
}

```
* Press enter to start the loop.
* When the calibration screen is ready. Start the stimulation procedure with Marius.
* As soon as the stimulated subject is inside, start the calibration procedure by pressing `c`. Karl/Marius will know how this works from here.
* When calibration is done. Press `escape` to start the actual task. The very first run there are still instructions. These can be skipped by pressing any key (except `q`), like the space bar. Usually the subject does it themselves.
* Now the script basically runs itself. Just:
  *  Use `q` to exit the script after the last volume has been colleceted
  *  Press `escape` when you see the calibration screen to exit the calibration phase and start the next run (unless you want to recalibrate, but during TMS you never want this).

* After the last (6th) run, immediately run `python payout.py <subject> <session> --settings 3t` to show the subject how much money they made on this session. This should also be noted down on the 1st sheeft of the Google spreadsheet document.
* Write TMS stimulation parameters down in the third sheet fo the Google spreadsheet document.




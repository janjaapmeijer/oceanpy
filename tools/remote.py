import os
import paramiko
import getpass
# password = getpass.getpass()
def ssh_cmd(command, hostname, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(hostname, username=username, password=password,
               allow_agent=False, look_for_keys=False)

    # channel = client.get_transport().open_session()
    stdin, stdout, stderr = client.exec_command(command)

    exit_status = stdout.channel.recv_exit_status()          # Blocking call
    if exit_status == 0:
        print(''.join(stdout.readlines()))
    else:
        print('Error',  exit_status)
        print(''.join(stderr.readlines()))
    client.close()

def sftp_copy_dir(localpath, remotepath, hostname, username, password):
    transport = paramiko.Transport((hostname, 22))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    os.chdir(os.path.split(localpath)[0])
    parent=os.path.split(localpath)[1]
    for walker in os.walk(parent):
        try:
            sftp.mkdir(remotepath.replace('\\','/'))
        except:
            pass
        try:
            sftp.mkdir(os.path.join(remotepath,walker[0]).replace('\\','/'))
        except:
            pass
        for file in walker[2]:
            sftp.put(os.path.join(walker[0],file),os.path.join(remotepath,walker[0],file).replace('\\','/'))
    sftp.close()
    transport.close()

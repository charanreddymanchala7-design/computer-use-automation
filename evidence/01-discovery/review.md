Look up a member and read their savings balance
member_lookup 1.0.0  [draft]  sha256:876701d613ef
Target: Fictional Systems Ltd MemberServ >=3.1,<4 (tenant profile default, surface web)

Sign in, find a member by number and read the current share savings balance. Read-only: nothing
is changed.

Needs
  input member_id (string, pattern ^[0-9]{5}$): Member number (five digits)
  secret MOCK_PASS (read from the environment, never stored)
  secret MOCK_USER (read from the environment, never stored)

Returns
  savings_balance (string): Current share savings balance, exactly as displayed, for example $2,480.15

Steps
  s1 [read] navigate: open the start page
      goes to /msv/login.cgi
  s2 [read] fill: Enter user id
      types secret MOCK_USER
      finds: input 'User ID' in frame top by ancestor_anchor > attribute_fingerprint
  s3 [read] fill: Enter password
      types secret MOCK_PASS
      finds: input 'Password' in frame top by ancestor_anchor > attribute_fingerprint
  s4 [read] click: Click login button
      finds: input '' in frame top by attribute_fingerprint
  s5 [read] fill: Enter member number
      types input member_id
      finds: input 'Member No:' in frame main by ancestor_anchor > attribute_fingerprint
  s6 [read] click: Click Go to search member
      finds: img 'Go' in frame main by attribute_fingerprint
  s7 [read] click: Click on member row to open member details
      finds: tr containing {member_id} in frame main by text
  s8 [read] extract: Extract the current share savings balance for member {member_id}
      returns savings_balance
      finds: value near 'SHARE SAVINGS' in frame main > acct by ancestor_anchor

Checkpoint (the run succeeded if all of these hold)
  'Member: {member_id}'; 'SHARE SAVINGS'; 'Status:'; url /msv/frameset.cgi

Handles
  no_such_member: business outcome; when 'NO RECORDS FOUND FOR CRITERIA', returns outcome member_not_found
  session_expired: hard failure; when 'User ID', fails with session_expired and brings in a human
  eod_notice: recoverable; when 'SYSTEM NOTICE: END-OF-DAY BATCH', recovers by dismiss (up to 2 tries)
  app_error: hard failure; when 'APPLICATION ERROR 0x8004', fails with app_error

Safety
  read-only: no step changes anything

Discovered by gemini-3.5-flash-lite in run 01-discovery (8 steps)

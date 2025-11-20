<!DOCTYPE html>
<html>
<body>
<h3>Live Stream</h3>
<img id="video" width="640">

<script>
let socket = new WebSocket("ws://" + location.hostname + ":8765");

socket.binaryType = "arraybuffer";

socket.onmessage = function(event) {
    let bytes = new Uint8Array(event.data);
    let blob = new Blob([bytes], {type: 'image/jpeg'});
    document.getElementById("video").src = URL.createObjectURL(blob);
};
</script>
</body>
</html>

// uygulama hakkında kısmı için

console.log("calisiyor");


//////////////////// UYGULAMA AÇIKLAMA İÇİN

var navappbtn = document.getElementById('about-app-btn');

navappbtn.onclick = function () {
	var appdiv = document.getElementById('about-app-id');

	if (appdiv.style.display !== 'none') {
		appdiv.style.display = 'none';
	}
	else {
		appdiv.style.display = 'block';
	}
}

var appclsbtn = document.getElementById('cls-btn-app-id')

appclsbtn.onclick = function () {
	var appdiv = document.getElementById('about-app-id');

	if (appdiv.style.display !== 'none') {
		appdiv.style.display = 'none';
	}
	else {
		appdiv.style.display = 'block';
	}
}

//////////////////// HOSPİTAL AÇIKLAMA İÇİN

var navhospitalbtn = document.getElementById('about-hospital-btn');

navhospitalbtn.onclick = function () {
	var hospitaldiv = document.getElementById('about-hospital-id');

	if (hospitaldiv.style.display !== 'none') {
		hospitaldiv.style.display = 'none';
	}
	else {
		hospitaldiv.style.display = 'block';
	}
}

var hospitalclsbtn = document.getElementById('cls-btn-hospital-id')

hospitalclsbtn.onclick = function () {
	var hospitaldiv = document.getElementById('about-hospital-id');

	if (hospitaldiv.style.display !== 'none') {
		hospitaldiv.style.display = 'none';
	}
	else {
		hospitaldiv.style.display = 'block';
	}
}
/////// İLETİŞİMM İÇİN
var navappbtn = document.getElementById('about-iletisim-btn');

navappbtn.onclick = function () {
	var appdiv = document.getElementById('about-iletisim-id');

	if (appdiv.style.display !== 'none') {
		appdiv.style.display = 'none';
	}
	else {
		appdiv.style.display = 'block';
	}
}

var appclsbtn = document.getElementById('cls-btn-iletisim-id')

appclsbtn.onclick = function () {
	var appdiv = document.getElementById('about-iletisim-id');

	if (appdiv.style.display !== 'none') {
		appdiv.style.display = 'none';
	}
	else {
		appdiv.style.display = 'block';
	}
}
